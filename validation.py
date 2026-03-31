import os
import json
import csv

# 直接使用你指定的絕對路徑，確保準確性
OUTPUT_DIR = r"D:\MKZ_Card_Lab\mkzcardlab_project\Wiwynn_project\test_data"
JSON_PATH  = os.path.join(OUTPUT_DIR, "ground_truth_segments.json")
CSV_PATH   = os.path.join(OUTPUT_DIR, "ground_truth_boxes.csv")

# 更新為你要保留的 33 張圖片 ID
KEEP = {
    "000000057777",
    "000000074883",
    "000000076195",
    "000000077396",
    "000000090964",
    "000000091006",
    "000000091885",
    "000000092729",
    "000000115378",
    "000000125690",
    "000000142454",
    "000000155749",
    "000000173345",
    "000000173814",
    "000000180738",
    "000000189504",
    "000000223932",
    "000000239654",
    "000000256481",
    "000000264473",
    "000000279809",
    "000000312550",
    "000000320867",
    "000000321980",
    "000000361217",
    "000000369796",
    "000000398810",
    "000000402473",
    "000000413395",
    "000000486536",
    "000000492810",
    "000000555705",
    "000000551815",
}

# 轉成有副檔名的 set
keep_files = {f"{name}.jpg" for name in KEEP}

# ── 過濾 JSON ────────────────────────────────────────────────────────────────
with open(JSON_PATH, "r", encoding="utf-8") as f:
    segments = json.load(f)

filtered_segments = {k: v for k, v in segments.items() if k in keep_files}

with open(JSON_PATH, "w", encoding="utf-8") as f:
    json.dump(filtered_segments, f, ensure_ascii=False, indent=2)

print(f"JSON：保留 {len(filtered_segments)} 張，移除 {len(segments) - len(filtered_segments)} 張")

# ── 過濾 CSV ─────────────────────────────────────────────────────────────────
with open(CSV_PATH, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    all_rows = list(reader)
    fieldnames = reader.fieldnames

kept_rows    = [r for r in all_rows if r["image_file"] in keep_files]
removed_rows = len(all_rows) - len(kept_rows)

with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(kept_rows)

print(f"CSV：保留 {len(kept_rows)} 行，移除 {removed_rows} 行")

# ── 刪除圖片 ─────────────────────────────────────────────────────────────────
all_imgs     = [f for f in os.listdir(OUTPUT_DIR) if f.endswith(".jpg")]
deleted      = 0
for img in all_imgs:
    if img not in keep_files:
        os.remove(os.path.join(OUTPUT_DIR, img))
        deleted += 1

print(f"圖片：保留 {len(keep_files)} 張，刪除 {deleted} 張")
print(f"\n✅ 完成")