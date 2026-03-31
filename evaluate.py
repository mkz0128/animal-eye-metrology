import json
import base64
import zlib
import numpy as np
import requests
import os
import cv2
from PIL import Image

IMG_DIR          = r"D:\MKZ_Card_Lab\mkzcardlab_project\Wiwynn_project\test_data"
GT_SEGMENTS_FILE = r"D:\MKZ_Card_Lab\mkzcardlab_project\Wiwynn_project\test_data\ground_truth_segments.json"
GT_EYES_FILE     = r"D:\MKZ_Card_Lab\mkzcardlab_project\Wiwynn_project\test_data\ground_truth_eyes.json"
GT_DIST_FILE     = r"D:\MKZ_Card_Lab\mkzcardlab_project\Wiwynn_project\test_data\ground_truth_distances.json"
API_MEASURE      = "http://localhost:8000/measure"
API_VISUALIZE    = "http://localhost:8000/visualize"

with open(GT_SEGMENTS_FILE, "r", encoding="utf-8") as f:
    gt_segments = json.load(f)
with open(GT_EYES_FILE, "r", encoding="utf-8") as f:
    gt_eyes = json.load(f)
with open(GT_DIST_FILE, "r", encoding="utf-8") as f:
    gt_distances = json.load(f)

def calc_dist(p1, p2):
    return float(np.linalg.norm(np.array(p1) - np.array(p2)))

def polygon_to_mask(polygon, img_w, img_h):
    mask = np.zeros((img_h, img_w), dtype=np.uint8)
    pts  = np.array(polygon, dtype=np.int32)
    cv2.fillPoly(mask, [pts], 1)
    return mask.astype(bool)

def compute_iou_mask(mask_a, mask_b):
    inter = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    return inter / union if union > 0 else 0.0

def rle_to_mask(rle):
    compressed = base64.b64decode(rle["data"])
    mask_bytes  = zlib.decompress(compressed)
    h, w        = rle["shape"]
    bits        = np.unpackbits(np.frombuffer(mask_bytes, dtype=np.uint8))
    return bits[:h * w].reshape(h, w).astype(bool)

EYE_MATCH_THRESHOLD = 10  # px，GT 與 Pred 眼睛座標距離門檻

animal_iou_list  = []
eye_mae_list     = []
intra_mae_list   = []
intra_mape_list  = []
inter_mae_list   = []
inter_mape_list  = []

gt_animal_count_list   = []
pred_animal_count_list = []
gt_eye_count_list      = []
pred_eye_count_list    = []

# 混淆矩陣累計
animal_tp_total = 0
animal_fp_total = 0
animal_fn_total = 0
eye_tp_total    = 0
eye_fp_total    = 0
eye_fn_total    = 0

skip_count = 0

filenames = sorted([f for f in os.listdir(IMG_DIR) if f.endswith(".jpg")])

for filename in filenames:
    img_path = os.path.join(IMG_DIR, filename)

    try:
        with open(img_path, "rb") as f:
            resp = requests.post(API_MEASURE, files={"file": f}, timeout=60)
        if resp.status_code != 200:
            print(f"❌ API 錯誤：{filename}")
            skip_count += 1
            continue
        pred = resp.json()
    except Exception as e:
        print(f"❌ 連線錯誤：{filename}  {e}")
        skip_count += 1
        continue

    img_pil = Image.open(img_path)
    img_w, img_h = img_pil.size

    pred_animals = pred["animals"]

    iou_vals   = []
    eye_mae    = None
    intra_vals = []
    inter_val  = None

    # ── 動物偵測統計 ──────────────────────────────────────────
    gt_animal_count   = len(gt_segments.get(filename, []))
    pred_animal_count = len(pred_animals)
    gt_animal_count_list.append(gt_animal_count)
    pred_animal_count_list.append(pred_animal_count)

    # ── 眼睛偵測統計 ──────────────────────────────────────────
    if filename in gt_eyes:
        gt_eye_count = len(gt_eyes[filename]["right_eye"]) + len(gt_eyes[filename]["left_eye"])
    else:
        gt_eye_count = 0
    pred_eye_count = sum(a["eyes_detected"] for a in pred_animals)
    gt_eye_count_list.append(gt_eye_count)
    pred_eye_count_list.append(pred_eye_count)

    # ── 動物偵測 IoU + 混淆矩陣 ──────────────────────────────
    animal_tp = 0
    animal_fn = 0
    matched_pred_animal = set()

    if filename in gt_segments:
        gt_cats = gt_segments[filename]

        for gt_cat in gt_cats:
            gt_poly = gt_cat["segmentation"]
            if not gt_poly or not pred_animals:
                animal_fn += 1
                continue
            gt_mask  = polygon_to_mask(gt_poly, img_w, img_h)
            best_iou = 0.0
            best_idx = -1

            for pi, pa in enumerate(pred_animals):
                if pi in matched_pred_animal:
                    continue
                if pa.get("mask_rle"):
                    pred_mask = rle_to_mask(pa["mask_rle"])
                else:
                    pred_mask = np.zeros((img_h, img_w), dtype=bool)
                    x1, y1, x2, y2 = int(pa["box"][0]), int(pa["box"][1]), int(pa["box"][2]), int(pa["box"][3])
                    pred_mask[y1:y2, x1:x2] = True

                iou = compute_iou_mask(gt_mask, pred_mask)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = pi

            if best_idx >= 0 and best_iou >= 0.5:
                matched_pred_animal.add(best_idx)
                animal_tp += 1
            else:
                animal_fn += 1

            iou_vals.append(round(best_iou, 3))
            animal_iou_list.append(best_iou)

    # FP：pred 偵測到但沒有配對到任何 GT
    animal_fp = sum(1 for pi in range(len(pred_animals)) if pi not in matched_pred_animal)

    animal_tp_total += animal_tp
    animal_fp_total += animal_fp
    animal_fn_total += animal_fn

    # ── 眼睛定位 MAE + 混淆矩陣 ──────────────────────────────
    eye_tp = 0
    eye_fn = 0
    matched_pred_eyes = set()  # 每個 pred 眼睛只能被配一次

    if filename in gt_eyes:
        gt_all = gt_eyes[filename]["right_eye"] + gt_eyes[filename]["left_eye"]
        pred_eyes_all = [
            (eye["center"], gi, ei)
            for gi, animal in enumerate(pred_animals)
            for ei, eye in enumerate(animal["eyes"])
        ]

        for gt_pt in gt_all:
            # 找距離在門檻內且還沒被配走的最近 pred 眼睛
            candidates = [
                (calc_dist(gt_pt, center), idx, center)
                for idx, (center, gi, ei) in enumerate(pred_eyes_all)
                if idx not in matched_pred_eyes
            ]
            candidates = [(d, idx, c) for d, idx, c in candidates if d <= EYE_MATCH_THRESHOLD]

            if candidates:
                best = min(candidates, key=lambda x: x[0])
                matched_pred_eyes.add(best[1])
                eye_tp += 1
            else:
                eye_fn += 1

        # 計算 MAE（不限門檻，只找最近的）
        pred_centers = [center for center, gi, ei in pred_eyes_all]
        if gt_all and pred_centers:
            errors = [min(calc_dist(gt_pt, p) for p in pred_centers) for gt_pt in gt_all]
            eye_mae = round(float(np.mean(errors)), 2)
            eye_mae_list.append(eye_mae)

    # FP：pred 眼睛沒有配對到任何 GT
    eye_fp = len(pred_eyes_all) - len(matched_pred_eyes) if filename in gt_eyes else 0

    eye_tp_total += eye_tp
    eye_fp_total += eye_fp
    eye_fn_total += eye_fn

    # ── 距離量測（層層依賴）────────────────────────────────────
    if filename in gt_distances:
        gt_dist = gt_distances[filename]

        pred_animal_rep_eyes = []
        for pi, pa in enumerate(pred_animals):
            if pa["eyes"]:
                pred_animal_rep_eyes.append((pa["eyes"][0]["center"], pi))

        pred_intra_by_idx = {
            p["animal_index"]: p["distance_px"]
            for p in pred["measurements"]["intra_animal_distances"]
        }

        all_intra_success  = True
        matched_pred_intra = set()

        for gt_d in gt_dist["intra_distances"]:
            gt_val       = gt_d["distance_px"]
            gt_right_eye = gt_d["right_eye"]
            if gt_val == 0:
                all_intra_success = False
                continue

            available = [(center, pi) for center, pi in pred_animal_rep_eyes
                         if pi not in matched_pred_intra]

            if not available:
                print(f"  ⚠️  IntraEye：{filename} 沒有可用的 pred 動物，跳過")
                all_intra_success = False
                continue

            best_pred_idx = min(available, key=lambda x: calc_dist(gt_right_eye, x[0]))[1]
            matched_pred_intra.add(best_pred_idx)

            pred_val = pred_intra_by_idx.get(best_pred_idx)
            if pred_val is None:
                print(f"  ⚠️  IntraEye：{filename} → Pred[{best_pred_idx}] 眼睛不足，跳過")
                all_intra_success = False
                continue

            mae  = abs(gt_val - pred_val)
            mape = mae / gt_val * 100
            intra_vals.append((gt_val, pred_val, round(mae, 2), round(mape, 2)))
            intra_mae_list.append(mae)
            intra_mape_list.append(mape)

        inter_list = gt_dist.get("inter_right_eye_dists", [])
        gt_inter = inter_list[0]["distance_px"] if inter_list else None

        if not all_intra_success:
            inter_val = None
        elif not gt_inter or gt_inter <= 0:
            inter_val = None
        else:
            pred_inter_list = pred["measurements"].get("inter_animal_distances", [])
            if not pred_inter_list:
                inter_val = None
            else:
                pred_inter_vals = [p["distance_px"] for p in pred_inter_list]
                closest_inter   = min(pred_inter_vals, key=lambda v: abs(v - gt_inter))
                mae  = abs(gt_inter - closest_inter)
                mape = mae / gt_inter * 100
                inter_val = (gt_inter, closest_inter, round(mae, 2), round(mape, 2))
                inter_mae_list.append(mae)
                inter_mape_list.append(mape)

    # ── 每張圖結果 ────────────────────────────────────────────
    print(f"\n{'='*50}")
    print(f"📷 {filename}")
    print(f"  動物：GT={gt_animal_count} Pred={pred_animal_count}  TP={animal_tp} FP={animal_fp} FN={animal_fn}")
    print(f"  眼睛：GT={gt_eye_count} Pred={pred_eye_count}  TP={eye_tp} FP={eye_fp} FN={eye_fn}")
    print(f"  動物偵測 IoU:   {iou_vals}")
    print(f"  眼睛定位 MAE:   {eye_mae} px")
    for gt_v, pred_v, mae, mape in intra_vals:
        print(f"  雙眼距離  GT:{gt_v}px  Pred:{pred_v}px  MAE:{mae}px  MAPE:{mape}%")
    if inter_val:
        gt_v, pred_v, mae, mape = inter_val
        print(f"  跨動物右眼 GT:{gt_v}px  Pred:{pred_v}px  MAE:{mae}px  MAPE:{mape}%")
    else:
        print(f"  跨動物右眼：跳過")

# ── 總結 ──────────────────────────────────────────────────────
animal_precision = animal_tp_total / (animal_tp_total + animal_fp_total) if (animal_tp_total + animal_fp_total) > 0 else 0
animal_recall    = animal_tp_total / (animal_tp_total + animal_fn_total) if (animal_tp_total + animal_fn_total) > 0 else 0
animal_f1        = 2 * animal_precision * animal_recall / (animal_precision + animal_recall) if (animal_precision + animal_recall) > 0 else 0

eye_precision = eye_tp_total / (eye_tp_total + eye_fp_total) if (eye_tp_total + eye_fp_total) > 0 else 0
eye_recall    = eye_tp_total / (eye_tp_total + eye_fn_total) if (eye_tp_total + eye_fn_total) > 0 else 0
eye_f1        = 2 * eye_precision * eye_recall / (eye_precision + eye_recall) if (eye_precision + eye_recall) > 0 else 0

print(f"\n{'='*60}")
print(f"📊 評估總結")
print(f"{'='*60}")
print(f"處理圖片數：{len(filenames) - skip_count}  跳過：{skip_count}")
print()
print(f"【動物偵測數量】")
print(f"  GT 總貓數：     {sum(gt_animal_count_list)}")
print(f"  Pred 總貓數：   {sum(pred_animal_count_list)}")
print(f"  平均 GT 貓數：  {round(np.mean(gt_animal_count_list), 2)}")
print(f"  平均 Pred 貓數：{round(np.mean(pred_animal_count_list), 2)}")
print()
print(f"【動物偵測混淆矩陣】（IoU ≥ 0.5）")
print(f"  TP={animal_tp_total}  FP={animal_fp_total}  FN={animal_fn_total}")
print(f"  Precision：{round(animal_precision * 100, 1)}%")
print(f"  Recall：   {round(animal_recall * 100, 1)}%")
print(f"  F1：       {round(animal_f1 * 100, 1)}%")
print()
print(f"【眼睛偵測數量】")
print(f"  GT 總眼睛數：     {sum(gt_eye_count_list)}")
print(f"  Pred 總眼睛數：   {sum(pred_eye_count_list)}")
print(f"  平均 GT 眼睛數：  {round(np.mean(gt_eye_count_list), 2)}")
print(f"  平均 Pred 眼睛數：{round(np.mean(pred_eye_count_list), 2)}")
print()
print(f"【眼睛偵測混淆矩陣】（距離門檻 ≤ {EYE_MATCH_THRESHOLD}px）")
print(f"  TP={eye_tp_total}  FP={eye_fp_total}  FN={eye_fn_total}")
print(f"  Precision：{round(eye_precision * 100, 1)}%")
print(f"  Recall：   {round(eye_recall * 100, 1)}%")
print(f"  F1：       {round(eye_f1 * 100, 1)}%")
print()
print(f"【動物偵測 IoU】")
print(f"  平均 IoU：{round(float(np.mean(animal_iou_list)), 3) if animal_iou_list else 'N/A'}")
print(f"  IoU > 0.5 比例：{round(sum(1 for v in animal_iou_list if v > 0.5) / len(animal_iou_list) * 100, 1) if animal_iou_list else 'N/A'}%")
print()
print(f"【眼睛定位】")
print(f"  平均 MAE：{round(float(np.mean(eye_mae_list)), 2) if eye_mae_list else 'N/A'} px")
print()
print(f"【雙眼距離量測】")
print(f"  平均 MAE： {round(float(np.mean(intra_mae_list)), 2) if intra_mae_list else 'N/A'} px")
print(f"  平均 MAPE：{round(float(np.mean(intra_mape_list)), 2) if intra_mape_list else 'N/A'} %")
print()
print(f"【跨動物右眼距離】")
print(f"  平均 MAE： {round(float(np.mean(inter_mae_list)), 2) if inter_mae_list else 'N/A'} px")
print(f"  平均 MAPE：{round(float(np.mean(inter_mape_list)), 2) if inter_mape_list else 'N/A'} %")

output = {
    "summary": {
        "total_images":       len(filenames) - skip_count,
        "skipped":            skip_count,
        "gt_total_animals":   int(sum(gt_animal_count_list)),
        "pred_total_animals": int(sum(pred_animal_count_list)),
        "gt_total_eyes":      int(sum(gt_eye_count_list)),
        "pred_total_eyes":    int(sum(pred_eye_count_list)),
    },
    "animal_detection": {
        "mean_iou":            round(float(np.mean(animal_iou_list)), 3) if animal_iou_list else None,
        "iou_above_0.5_ratio": round(sum(1 for v in animal_iou_list if v > 0.5) / len(animal_iou_list), 3) if animal_iou_list else None,
        "tp": animal_tp_total,
        "fp": animal_fp_total,
        "fn": animal_fn_total,
        "precision": round(animal_precision, 3),
        "recall":    round(animal_recall, 3),
        "f1":        round(animal_f1, 3),
    },
    "eye_localization": {
        "mean_mae_px":    round(float(np.mean(eye_mae_list)), 2) if eye_mae_list else None,
        "match_threshold_px": EYE_MATCH_THRESHOLD,
        "tp": eye_tp_total,
        "fp": eye_fp_total,
        "fn": eye_fn_total,
        "precision": round(eye_precision, 3),
        "recall":    round(eye_recall, 3),
        "f1":        round(eye_f1, 3),
    },
    "intra_animal_distance": {
        "mean_mae_px":   round(float(np.mean(intra_mae_list)), 2) if intra_mae_list else None,
        "mean_mape_pct": round(float(np.mean(intra_mape_list)), 2) if intra_mape_list else None
    },
    "inter_animal_distance": {
        "mean_mae_px":   round(float(np.mean(inter_mae_list)), 2) if inter_mae_list else None,
        "mean_mape_pct": round(float(np.mean(inter_mape_list)), 2) if inter_mape_list else None
    }
}

with open("evaluation_results.json", "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=2)

print(f"\n✅ 結果存至 evaluation_results.json")