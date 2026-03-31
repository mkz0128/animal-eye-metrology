import os
import io
import base64
import zlib
import torch
import cv2
import numpy as np
from itertools import combinations
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
from PIL import Image, ImageEnhance
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from contextlib import asynccontextmanager

processor = None
dino_model = None
sam2_predictor = None
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DINO_MODEL_ID = "IDEA-Research/grounding-dino-base"
SAM2_CHECKPOINT = "/app/checkpoints/sam2.1_hiera_tiny.pt"
SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_t.yaml"

CAT_THRESHOLD = 0.15
CAT_NMS_IOU = 0.4
CAT_CONTAIN_RATIO = 0.7
EYE_THRESHOLD = 0.10
EYE_NMS_IOU = 0.3
EYE_MAX_SIZE_RATIO = 0.2
PAIR_MIN_DIST_RATIO = 0.05
PAIR_MAX_DIST_RATIO = 0.35
PAIR_MAX_AREA_RATIO = 4.0
EYE_UPSCALE_TARGET = 512


@asynccontextmanager
async def lifespan(app):
    global processor, dino_model, sam2_predictor
    print(f"正在載入模型至 {DEVICE}...")
    processor = AutoProcessor.from_pretrained(DINO_MODEL_ID)
    dino_model = AutoModelForZeroShotObjectDetection.from_pretrained(DINO_MODEL_ID).to(DEVICE)
    sam2_model = build_sam2(SAM2_CONFIG, SAM2_CHECKPOINT, device=DEVICE)
    sam2_predictor = SAM2ImagePredictor(sam2_model)
    print("所有模型載入完成！")
    yield


app = FastAPI(title="Animal Eye Metrology API", lifespan=lifespan)


def calculate_distance(p1, p2):
    return float(np.linalg.norm(np.array(p1) - np.array(p2)))

def get_mask_center(mask):
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    return [float(np.mean(xs)), float(np.mean(ys))]

def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0

def get_box_area(box):
    return max(0, box[2] - box[0]) * max(0, box[3] - box[1])

def nms_by_iou(detections, iou_threshold=0.5):
    if len(detections) <= 1:
        return detections
    dets = sorted(detections, key=lambda d: d["score"], reverse=True)
    keep = []
    while dets:
        best = dets.pop(0)
        keep.append(best)
        dets = [d for d in dets if compute_iou(best["box"], d["box"]) < iou_threshold]
    return keep

def filter_contained_boxes(detections, contain_ratio=CAT_CONTAIN_RATIO):
    if len(detections) <= 1:
        return detections
    keep = []
    for i, d in enumerate(detections):
        contained = False
        area_d = get_box_area(d["box"])
        if area_d == 0:
            continue
        for j, other in enumerate(detections):
            if i == j:
                continue
            x1 = max(d["box"][0], other["box"][0])
            y1 = max(d["box"][1], other["box"][1])
            x2 = min(d["box"][2], other["box"][2])
            y2 = min(d["box"][3], other["box"][3])
            inter = max(0, x2 - x1) * max(0, y2 - y1)
            if inter / area_d > contain_ratio:
                contained = True
                break
        if not contained:
            keep.append(d)
    return keep

def mask_to_rle(mask):
    mask_bytes = np.packbits(mask.flatten()).tobytes()
    compressed = zlib.compress(mask_bytes)
    return {
        "data": base64.b64encode(compressed).decode("utf-8"),
        "shape": list(mask.shape)
    }

def detect_objects(pil_img, text_prompt, threshold=0.35, text_threshold=0.25):
    inputs = processor(images=pil_img, text=text_prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad(), torch.autocast(device_type="cuda", enabled=(DEVICE == "cuda")):
        outputs = dino_model(**inputs)
    w, h = pil_img.size
    results = processor.post_process_grounded_object_detection(
        outputs, input_ids=inputs["input_ids"],
        threshold=threshold, text_threshold=text_threshold,
        target_sizes=[(h, w)]
    )[0]
    detections = []
    for box, score, label in zip(results["boxes"], results["scores"], results["text_labels"]):
        b = box.tolist()
        center = [(b[0] + b[2]) / 2, (b[1] + b[3]) / 2]
        detections.append({"box": b, "center": center, "score": round(score.item(), 3), "label": label})
    return detections

def get_sam2_masks(image_np, boxes):
    if len(boxes) == 0:
        return []
    sam2_predictor.set_image(image_np)
    masks_list = []
    for box in np.array(boxes):
        masks, scores, _ = sam2_predictor.predict(box=box, multimask_output=False)
        masks_list.append(masks[0].astype(bool))
    return masks_list

def is_valid_eye_pair(eye1, eye2, crop_w):
    dist = calculate_distance(eye1["center"], eye2["center"])
    if dist < crop_w * PAIR_MIN_DIST_RATIO or dist > crop_w * PAIR_MAX_DIST_RATIO:
        return False
    area1 = get_box_area(eye1["box"])
    area2 = get_box_area(eye2["box"])
    if area1 > 0 and area2 > 0:
        if max(area1, area2) / min(area1, area2) > PAIR_MAX_AREA_RATIO:
            return False
    return True

def find_best_eye_pair(detections, crop_w):
    best_pair = None
    best_score = -1
    for i in range(len(detections)):
        for j in range(i + 1, len(detections)):
            if not is_valid_eye_pair(detections[i], detections[j], crop_w):
                continue
            score = (detections[i]["score"] + detections[j]["score"]) / 2
            area1 = get_box_area(detections[i]["box"])
            area2 = get_box_area(detections[j]["box"])
            if area1 > 0 and area2 > 0:
                score += min(area1, area2) / max(area1, area2) * 0.1
            if score > best_score:
                best_score = score
                best_pair = [detections[i], detections[j]]
    return best_pair

def detect_eyes_for_animal(pil_img, img_rgb, animal_box):
    b = animal_box
    pad = 10
    x1 = max(0, int(b[0]) - pad)
    y1 = max(0, int(b[1]) - pad)
    x2 = min(pil_img.width,  int(b[2]) + pad)
    y2 = min(pil_img.height, int(b[3]) + pad)
    crop_w = x2 - x1
    crop_h = y2 - y1
    crop_np = img_rgb[y1:y2, x1:x2]
    crop = pil_img.crop((x1, y1, x2, y2))

    crop_enhanced = ImageEnhance.Contrast(crop).enhance(1.3)
    crop_enhanced = ImageEnhance.Sharpness(crop_enhanced).enhance(1.2)
    arr = np.array(crop_enhanced)
    lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    arr = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    crop_enhanced = Image.fromarray(arr)

    scale = EYE_UPSCALE_TARGET / max(crop_w, crop_h)
    if scale > 1.0:
        new_w = int(crop_w * scale)
        new_h = int(crop_h * scale)
        crop_for_dino = crop_enhanced.resize((new_w, new_h), Image.LANCZOS)
    else:
        scale = 1.0
        crop_for_dino = crop_enhanced

    all_dets = detect_objects(crop_for_dino, "cat eye. cat ear. cat nose. cat mouth.", threshold=EYE_THRESHOLD)
    all_dets = nms_by_iou(all_dets, iou_threshold=EYE_NMS_IOU)

    for d in all_dets:
        d["box"] = [v / scale for v in d["box"]]
        d["center"] = [(d["box"][0] + d["box"][2]) / 2, (d["box"][1] + d["box"][3]) / 2]

    eye_dets = [d for d in all_dets if "eye" in d["label"]]
    eye_dets = [
        e for e in eye_dets
        if (e["box"][2] - e["box"][0]) < crop_w * EYE_MAX_SIZE_RATIO
        and (e["box"][3] - e["box"][1]) < crop_h * EYE_MAX_SIZE_RATIO
    ]

    if len(eye_dets) > 2:
        pair = find_best_eye_pair(eye_dets, crop_w)
        eye_dets = pair if pair else [max(eye_dets, key=lambda e: e["score"])]
    elif len(eye_dets) == 2:
        if not is_valid_eye_pair(eye_dets[0], eye_dets[1], crop_w):
            eye_dets = [max(eye_dets, key=lambda e: e["score"])]

    eye_boxes = [e["box"] for e in eye_dets]
    eye_masks = get_sam2_masks(crop_np, eye_boxes)

    eyes = []
    for j, eye in enumerate(eye_dets):
        if j < len(eye_masks):
            mc = get_mask_center(eye_masks[j])
            center = [mc[0] + x1, mc[1] + y1] if mc else [eye["center"][0] + x1, eye["center"][1] + y1]
            mask_full = np.zeros(img_rgb.shape[:2], dtype=bool)
            lm = eye_masks[j]
            mask_full[y1:y1 + lm.shape[0], x1:x1 + lm.shape[1]] = lm
        else:
            center = [eye["center"][0] + x1, eye["center"][1] + y1]
            mask_full = None
        eyes.append({"center": center, "score": eye["score"], "mask": mask_full})

    b = animal_box
    eyes = [
        e for e in eyes
        if b[0] <= e["center"][0] <= b[2] and b[1] <= e["center"][1] <= b[3]
    ]

    return eyes

def process_image(pil_img, img_rgb):
    cat_dets = detect_objects(pil_img, "cat.", threshold=CAT_THRESHOLD)
    cat_dets = nms_by_iou(cat_dets, iou_threshold=CAT_NMS_IOU)
    cat_dets = filter_contained_boxes(cat_dets)

    cat_boxes = [d["box"] for d in cat_dets]
    cat_masks = get_sam2_masks(img_rgb, cat_boxes)

    animals = []
    for i, det in enumerate(cat_dets):
        eyes = detect_eyes_for_animal(pil_img, img_rgb, det["box"])
        animals.append({
            "box": det["box"], "center": det["center"], "score": det["score"],
            "eyes": eyes,
            "mask": cat_masks[i] if i < len(cat_masks) else None
        })

    intra_distances = []
    for i, a in enumerate(animals):
        if len(a["eyes"]) >= 2:
            se = sorted(a["eyes"], key=lambda e: e["center"][0])
            d = calculate_distance(se[0]["center"], se[1]["center"])
            intra_distances.append({
                "animal_index": i, "distance_px": round(d, 2),
                "right_eye": se[0]["center"], "left_eye": se[1]["center"]
            })

    right_eyes = []
    for i, a in enumerate(animals):
        if len(a["eyes"]) >= 2:
            right_eye = min(a["eyes"], key=lambda e: e["center"][0])
            right_eyes.append({"animal_index": i, "center": right_eye["center"]})

    inter_distances = []
    all_have_two_eyes = all(len(a["eyes"]) >= 2 for a in animals)

    if all_have_two_eyes and len(right_eyes) >= 2:
        for (a, b) in combinations(right_eyes, 2):
            d = calculate_distance(a["center"], b["center"])
            inter_distances.append({
                "animal_a": a["animal_index"], "animal_b": b["animal_index"],
                "distance_px": round(d, 2), "eye_a": a["center"], "eye_b": b["center"]
            })

    measurements = {
        "intra_animal_distances": intra_distances,
        "inter_animal_distances": inter_distances,
        "inter_animal_right_eye_dist": inter_distances[0]["distance_px"] if inter_distances else None,
        "inter_animal_right_eye_line": [inter_distances[0]["eye_a"], inter_distances[0]["eye_b"]] if inter_distances else None
    }

    return animals, measurements


@app.post("/measure")
async def measure_animal_eyes(file: UploadFile = File(...)):
    image_bytes = await file.read()
    nparr  = np.frombuffer(image_bytes, np.uint8)
    img_cv = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    animals, measurements = process_image(pil_img, img_rgb)
    return {
        "status": "success",
        "detected_animals": len(animals),
        "animals": [
            {
                "box": a["box"], "score": a["score"],
                "eyes_detected": len(a["eyes"]),
                "eyes": [{"center": e["center"], "score": e["score"]} for e in a["eyes"]],
                "has_segmentation": a["mask"] is not None,
                "mask_rle": mask_to_rle(a["mask"]) if a["mask"] is not None else None
            }
            for a in animals
        ],
        "measurements": {
            "intra_animal_distances":  measurements["intra_animal_distances"],
            "inter_animal_distances":  measurements["inter_animal_distances"],
            "inter_animal_right_eye_dist": measurements["inter_animal_right_eye_dist"]
        }
    }


@app.post("/visualize")
async def visualize(file: UploadFile = File(...)):
    image_bytes = await file.read()
    nparr  = np.frombuffer(image_bytes, np.uint8)
    img_cv = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    animals, measurements = process_image(pil_img, img_rgb)

    draw_img = img_cv.copy()
    colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0)]

    for i, animal in enumerate(animals):
        color = colors[i % len(colors)]
        if animal["mask"] is not None:
            overlay = draw_img.copy()
            overlay[animal["mask"]] = color
            draw_img = cv2.addWeighted(draw_img, 0.7, overlay, 0.3, 0)
            contours, _ = cv2.findContours(animal["mask"].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(draw_img, contours, -1, color, 2)

        b = animal["box"]
        cv2.putText(draw_img, f"Cat {i}", (int(b[0]), int(b[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        for eye in animal["eyes"]:
            if eye.get("mask") is not None:
                overlay = draw_img.copy()
                overlay[eye["mask"]] = (0, 0, 255)
                draw_img = cv2.addWeighted(draw_img, 0.6, overlay, 0.4, 0)
            cx, cy = int(eye["center"][0]), int(eye["center"][1])
            cv2.circle(draw_img, (cx, cy), 5, (0, 0, 255), -1)

        if len(animal["eyes"]) >= 2:
            se = sorted(animal["eyes"], key=lambda e: e["center"][0])
            p1 = (int(se[0]["center"][0]), int(se[0]["center"][1]))
            p2 = (int(se[1]["center"][0]), int(se[1]["center"][1]))
            cv2.line(draw_img, p1, p2, (0, 255, 255), 2)
            d = calculate_distance(se[0]["center"], se[1]["center"])
            mid = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2 - 10)
            cv2.putText(draw_img, f"Cat{i}: {d:.1f}px", mid, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    for inter in measurements.get("inter_animal_distances", []):
        p1 = (int(inter["eye_a"][0]), int(inter["eye_a"][1]))
        p2 = (int(inter["eye_b"][0]), int(inter["eye_b"][1]))
        cv2.line(draw_img, p1, p2, (255, 255, 255), 2)
        d = inter["distance_px"]
        mid = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2 - 15)
        cv2.putText(draw_img, f"Inter Cat{inter['animal_a']}-Cat{inter['animal_b']}: {d:.1f}px",
                    mid, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    _, buffer = cv2.imencode(".jpg", draw_img)
    return StreamingResponse(io.BytesIO(buffer.tobytes()), media_type="image/jpeg")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "cuda": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"
    }