import json
import numpy as np

CVAT_JSON = "instances_default.json"  # 你匯出的檔案路徑

with open(CVAT_JSON, "r", encoding="utf-8") as f:
    data = json.load(f)

# 建立 image_id -> filename 的對應
id_to_file = {img["id"]: img["file_name"] for img in data["images"]}

# 建立 category_id -> label 的對應
id_to_label = {cat["id"]: cat["name"] for cat in data["categories"]}

# 整理成 {filename: {right_eye: [...], left_eye: [...]}}
gt_eyes = {}
for ann in data["annotations"]:
    filename = id_to_file[ann["image_id"]]
    label = id_to_label[ann["category_id"]]
    
    # 從 segmentation polygon 計算中心點
    poly = ann["segmentation"][0]
    xs = poly[0::2]
    ys = poly[1::2]
    cx = round(sum(xs) / len(xs), 1)
    cy = round(sum(ys) / len(ys), 1)
    
    if filename not in gt_eyes:
        gt_eyes[filename] = {"right_eye": [], "left_eye": []}
    
    gt_eyes[filename][label].append([cx, cy])

# 存成乾淨的 JSON
output_path = "ground_truth_eyes.json"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(gt_eyes, f, ensure_ascii=False, indent=2)

print(f"✅ 完成，{len(gt_eyes)} 張圖")
print(f"✅ 存至 {output_path}")

# 印出前3張確認格式
for i, (fname, eyes) in enumerate(gt_eyes.items()):
    print(f"\n{fname}")
    print(f"  right_eye: {eyes['right_eye']}")
    print(f"  left_eye:  {eyes['left_eye']}")
    if i >= 2:
        break