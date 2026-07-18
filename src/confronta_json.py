"""
confronta_json.py -- side-by-side arbiter for two dictionary bundles that
claim the same identity. Prints identity fields, decoding, and the largest
cosine differences so you can decide which one is canonical.

Usage:  python confronta_json.py archivio/fileA.json archivio/fileB.json
"""
import json, sys

a, b = (json.load(open(p)) for p in sys.argv[1:3])
pa, pb = sys.argv[1], sys.argv[2]
print(f"{'':22s} A = {pa}\n{'':22s} B = {pb}\n")
for k in ("model", "k_relations", "pairs_per_relation", "seed", "peak_block",
          "write_layer", "revision", "folds", "gauge"):
    va, vb = a.get(k), b.get(k)
    mark = "   <-- DIFFERS" if va != vb else ""
    print(f"{k:22s} {str(va):34.34s} {str(vb):34.34s}{mark}")
da = a.get("decoding", {}).get("delta_f", {}).get("acc")
db = b.get("decoding", {}).get("delta_f", {}).get("acc")
print(f"{'decoding delta_f':22s} {str(da):34.34s} {str(db):34.34s}"
      + ("   <-- DIFFERS" if da != db else ""))
if a.get("cats") == b.get("cats") and "cos_peak" in a and "cos_peak" in b:
    cats = a["cats"]; K = len(cats)
    diffs = []
    for i in range(K):
        for j in range(i + 1, K):
            d = abs(a["cos_peak"][i][j] - b["cos_peak"][i][j])
            diffs.append((d, cats[i], cats[j], a["cos_peak"][i][j], b["cos_peak"][i][j]))
    diffs.sort(reverse=True)
    print(f"\nmax |Delta cos_peak| = {diffs[0][0]:.4f}   "
          f"({'identical matrices' if diffs[0][0] < 1e-9 else 'top differing cells below'})")
    for d, ci, cj, va, vb in diffs[:5]:
        if d < 1e-9: break
        print(f"  {ci}-{cj}: {va:+.3f} vs {vb:+.3f}   (|delta| {d:.3f})")
else:
    print("\n(cats differ or cos_peak missing: matrices not comparable)")
