import json
import numpy as np
from itertools import combinations

GT_FILE = r"D:\MKZ_Card_Lab\mkzcardlab_project\Wiwynn_project\test_data\ground_truth_eyes.json"

with open(GT_FILE, "r", encoding="utf-8") as f:
    gt_eyes = json.load(f)

def distance(p1, p2):
    return float(np.linalg.norm(np.array(p1) - np.array(p2)))

results = {}

for filename, eyes in gt_eyes.items():
    right_eyes = eyes.get("right_eye", [])
    left_eyes  = eyes.get("left_eye", [])
    
    entry = {
        "intra_distances": [],
        "inter_right_eye_dists": []
    }

    used_left = set()
    for r_idx, r_point in enumerate(right_eyes):
        min_dist = float('inf')
        best_l_idx = -1
        for l_idx, l_point in enumerate(left_eyes):
            if l_idx in used_left:
                continue
            d = distance(r_point, l_point)
            if d < min_dist:
                min_dist = d
                best_l_idx = l_idx
        
        if best_l_idx != -1:
            used_left.add(best_l_idx)
            entry["intra_distances"].append({
                "cat_index": r_idx,
                "right_eye": r_point,        # ← 加這行
                "distance_px": round(min_dist, 2)
            })

    if len(right_eyes) >= 2:
        for (i, p1), (j, p2) in combinations(enumerate(right_eyes), 2):
            d = distance(p1, p2)
            entry["inter_right_eye_dists"].append({
                "cats_pair": f"{i} and {j}",
                "distance_px": round(d, 2)
            })

    results[filename] = entry

    print(f"\n{filename}")
    for intra in entry["intra_distances"]:
        print(f"  Cat {intra['cat_index']} 雙眼距離: {intra['distance_px']} px")
    for inter in entry["inter_right_eye_dists"]:
        print(f"  Cat {inter['cats_pair']} 右眼距離: {inter['distance_px']} px")

out_path = r"D:\MKZ_Card_Lab\mkzcardlab_project\Wiwynn_project\test_data\ground_truth_distances.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print(f"\n✅ 存至 {out_path}")