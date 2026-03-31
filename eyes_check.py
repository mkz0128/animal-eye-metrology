import json
import os
from PIL import Image, ImageDraw

CVAT_JSON = r"D:\MKZ_Card_Lab\mkzcardlab_project\Wiwynn_project\test_data\instances_default.json"
IMG_DIR   = r"D:\MKZ_Card_Lab\mkzcardlab_project\Wiwynn_project\test_data"

with open(CVAT_JSON, "r", encoding="utf-8") as f:
    data = json.load(f)

id_to_file  = {img["id"]: img["file_name"] for img in data["images"]}
id_to_label = {cat["id"]: cat["name"] for cat in data["categories"]}

file_to_anns = {}
for ann in data["annotations"]:
    filename = id_to_file[ann["image_id"]]
    label    = id_to_label[ann["category_id"]]
    poly_flat = ann["segmentation"][0]
    polygon  = [
        (poly_flat[i], poly_flat[i+1])
        for i in range(0, len(poly_flat), 2)
    ]
    if filename not in file_to_anns:
        file_to_anns[filename] = []
    file_to_anns[filename].append({"label": label, "polygon": polygon})

filenames = list(file_to_anns.keys())
print(f"共 {len(filenames)} 張圖，按 Enter 看下一張，Ctrl+C 離開")

for idx, filename in enumerate(filenames):
    img_path = os.path.join(IMG_DIR, filename)
    if not os.path.exists(img_path):
        print(f"⚠️  找不到：{filename}")
        continue

    img     = Image.open(img_path).convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw    = ImageDraw.Draw(overlay)

    for ann in file_to_anns[filename]:
        poly  = ann["polygon"]
        label = ann["label"]

        if label == "right_eye":
            fill    = (255, 0, 0, 120)
            outline = (255, 0, 0)
        else:
            fill    = (0, 0, 255, 120)
            outline = (0, 0, 255)

        if len(poly) >= 3:
            draw.polygon(poly, fill=fill)
            draw.line(poly + [poly[0]], fill=outline, width=2)

        cx = sum(p[0] for p in poly) / len(poly)
        cy = sum(p[1] for p in poly) / len(poly)
        draw.text((cx+4, cy-8), label[0].upper(), fill=outline)

    result = Image.alpha_composite(img, overlay).convert("RGB")
    print(f"[{idx+1}/{len(filenames)}] {filename}  ({len(file_to_anns[filename])} 個標注)")
    result.show()
    input("按 Enter 看下一張，Ctrl+C 離開...")