import csv
import numpy as np

rows = list(csv.DictReader(open("final_test.csv")))
for r in rows:
    for k in ("n_holes", "c_exact", "c_occ"):
        r[k] = float(r[k])
    r["closed"] = r["closed"] == "True"

trusted = [r for r in rows if r["closed"] and r["n_holes"] <= 50]
untrusted = [r for r in rows if not (r["closed"] and r["n_holes"] <= 50)]

print(f"trusted: {len(trusted)}, untrusted: {len(untrusted)}")
print(f"occupancy compactness > 1: trusted {sum(r['c_occ'] > 1 for r in trusted)}, "
      f"untrusted {sum(r['c_occ'] > 1 for r in untrusted)}")
print(f"exact compactness > 1 among trusted: {sum(r['c_exact'] > 1 for r in trusted)}")

usable_fallback = [r for r in untrusted if r["c_occ"] <= 1]
print(f"untrusted meshes with a plausible occupancy value: {len(usable_fallback)} of {len(untrusted)}")

# Among trusted meshes where occupancy is plausible (<= 1), how well does it match the exact value?
ok_pairs = [abs(r["c_exact"] - r["c_occ"]) for r in trusted if r["c_occ"] <= 1]
print(f"trusted with occupancy <= 1: {len(ok_pairs)}, "
      f"median diff {np.median(ok_pairs):.3f}, 90th pct {np.percentile(ok_pairs, 90):.3f}")