import os
import json
import shutil

BASE_DIR  = r"D:\MKZ_Card_Lab\mkzcardlab_project\Wiwynn_project\test_data"
JSON_FILES = [
    os.path.join(BASE_DIR, "ground_truth_segments.json"),
    os.path.join(BASE_DIR, "ground_truth_eyes.json"),
    os.path.join(BASE_DIR, "ground_truth_distances.json"),
]

REMOVE = {
    "000000320867.jpg",
}

# 刪圖片
for filename in REMOVE:
    img_path = os.path.join(BASE_DIR, filename)
    if os.path.exists(img_path):
        os.remove(img_path)
        print(f"✅ 刪除圖片：{filename}")
    else:
        print(f"⚠️  找不到圖片：{filename}")

# 刪 JSON 資料
for json_path in JSON_FILES:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    before = len(data)
    data = {k: v for k, v in data.items() if k not in REMOVE}
    after = len(data)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"✅ {os.path.basename(json_path)}：移除 {before - after} 筆，剩 {after} 筆")

print("\n✅ 完成")