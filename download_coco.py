import os
import json
import csv
import shutil
import requests
from pycocotools.coco import COCO

# =============================================================================
# 設定
# =============================================================================
TRAIN_ANN_FILE  = r"C:\Users\USER\fiftyone\coco-2017\raw\instances_train2017.json"
TRAIN_IMG_CACHE = r"C:\Users\USER\fiftyone\coco-2017\train\data"
VAL_ANN_FILE    = r"C:\Users\USER\fiftyone\coco-2017\raw\instances_val2017.json"
VAL_IMG_CACHE   = r"C:\Users\USER\fiftyone\coco-2017\validation\data"
OUTPUT_DIR      = "test_data"
TRAIN_COUNT     = 200
VAL_COUNT       = 100

segments_dict = {}
csv_rows      = []

if os.path.exists(OUTPUT_DIR):
    shutil.rmtree(OUTPUT_DIR)
os.makedirs(OUTPUT_DIR)


# =============================================================================
# 共用函數
# =============================================================================
def process_split(ann_file, img_cache, target_count, split_name):
    print(f"\n{'='*50}")
    print(f"載入 {split_name} annotation...")
    coco = COCO(ann_file)
    cat_id = coco.getCatIds(catNms=["cat"])[0]

    print(f"篩選 {split_name} 圖片...")
    multi_cat = []
    for img_id in coco.getImgIds(catIds=[cat_id]):
        anns = coco.loadAnns(
            coco.getAnnIds(imgIds=img_id, catIds=[cat_id], iscrowd=False)
        )
        if len(anns) >= 2:
            multi_cat.append((img_id, anns))
        if len(multi_cat) >= target_count:
            break

    print(f"找到 {len(multi_cat)} 張符合條件的圖片")

    for idx, (img_id, anns) in enumerate(multi_cat):
        img_info = coco.loadImgs(img_id)[0]
        filename = img_info["file_name"]

        src = os.path.join(img_cache, filename)
        dst = os.path.join(OUTPUT_DIR, filename)
        if os.path.exists(src):
            shutil.copy(src, dst)
            source = "cache"
        else:
            r = requests.get(img_info["coco_url"], stream=True)
            with open(dst, "wb") as f:
                for chunk in r.iter_content(8192):
                    f.write(chunk)
            source = "download"

        segments_dict[filename] = []

        for i, ann in enumerate(anns):
            x, y, w, h = ann["bbox"]
            x1 = round(x, 1)
            y1 = round(y, 1)
            x2 = round(x + w, 1)
            y2 = round(y + h, 1)
            cx = round((x1 + x2) / 2, 1)
            cy = round((y1 + y2) / 2, 1)

            polygon_px = []
            if ann["segmentation"] and isinstance(ann["segmentation"], list):
                flat = ann["segmentation"][0]
                polygon_px = [
                    [round(flat[j], 1), round(flat[j+1], 1)]
                    for j in range(0, len(flat), 2)
                ]

            segments_dict[filename].append({
                "cat_index":    i,
                "split":        split_name,
                "bbox":         [x1, y1, x2, y2],
                "segmentation": polygon_px
            })

            csv_rows.append({
                "image_file":    filename,
                "split":         split_name,
                "cat_index":     i,
                "bbox_x1":       x1,
                "bbox_y1":       y1,
                "bbox_x2":       x2,
                "bbox_y2":       y2,
                "bbox_w":        round(w, 1),
                "bbox_h":        round(h, 1),
                "bbox_center_x": cx,
                "bbox_center_y": cy,
            })

        seg_counts = [
            len(segments_dict[filename][i]["segmentation"])
            for i in range(len(anns))
        ]
        print(f"  [{idx+1}/{target_count}] {filename}  ({len(anns)} 隻貓)  seg點數:{seg_counts}  [{source}]")


# =============================================================================
# 執行
# =============================================================================
process_split(TRAIN_ANN_FILE, TRAIN_IMG_CACHE, TRAIN_COUNT, "train")
process_split(VAL_ANN_FILE,   VAL_IMG_CACHE,   VAL_COUNT,   "val")

# 存 JSON
json_path = os.path.join(OUTPUT_DIR, "ground_truth_segments.json")
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(segments_dict, f, ensure_ascii=False, indent=2)

# 存 CSV
csv_path = os.path.join(OUTPUT_DIR, "ground_truth_boxes.csv")
with open(csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=[
        "image_file", "split", "cat_index",
        "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2",
        "bbox_w", "bbox_h", "bbox_center_x", "bbox_center_y"
    ])
    writer.writeheader()
    writer.writerows(csv_rows)

print(f"\n{'='*50}")
print(f"✅ 完成，共 {len(segments_dict)} 張圖片")
print(f"   train: {TRAIN_COUNT} 張，val: {VAL_COUNT} 張")
print(f"✅ Segmentation → {json_path}")
print(f"✅ Bounding box → {csv_path}")