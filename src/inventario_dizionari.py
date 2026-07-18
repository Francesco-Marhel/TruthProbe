"""
inventario_dizionari.py -- census and archive of every dictionary bundle.
v2: Windows-safe names, .pt inherits identity from its twin .json,
copy failures are reported instead of crashing.

Run ONCE from the studio root, no arguments:
    python inventario_dizionari.py
It scans the whole tree, reads the identity STORED INSIDE each file,
hashes contents, copies one copy of each unique file into ./archivio/
with a self-describing name, and writes inventario.csv. Originals are
never touched.
"""
import csv, hashlib, json, os, re, shutil, sys, time

ILLEGAL = re.compile(r'[<>:"/\\|?*]')

def sanitize(name):
    return ILLEGAL.sub("-", name)

def sha8(path):
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:8]

def read_bundle(path):
    if path.endswith(".json"):
        return json.load(open(path))
    import torch
    return torch.load(path, map_location="cpu")

def identity_from(path):
    try:
        d = read_bundle(path)
    except Exception as e:
        return None, f"unreadable ({type(e).__name__})"
    if "cats" not in d and "cos_peak" not in d:
        return None, "not a dictionary bundle"
    ident = dict(
        model=str(d.get("model", "model-x")).split("/")[-1],
        k=d.get("k_relations", len(d.get("cats", [])) or "x"),
        n=d.get("pairs_per_relation", None),
        seed=d.get("seed", None),
        gauge=bool(d.get("gauge")),
    )
    # a .pt with missing fields inherits them from its twin .json
    if path.endswith(".pt"):
        twin = os.path.splitext(path)[0] + ".json"
        if os.path.exists(twin):
            try:
                t = json.load(open(twin))
                if ident["model"] == "model-x":
                    ident["model"] = str(t.get("model", "model-x")).split("/")[-1]
                if ident["n"] is None:
                    ident["n"] = t.get("pairs_per_relation")
                if ident["seed"] is None:
                    ident["seed"] = t.get("seed")
                if ident["k"] == "x":
                    ident["k"] = t.get("k_relations", "x")
            except Exception:
                pass
    if ident["n"] is None: ident["n"] = "x"
    if ident["seed"] is None: ident["seed"] = "x"
    return ident, None

def main():
    roots = sys.argv[1:] or ["."]
    found = []
    for root in roots:
        for dirpath, _dirs, files in os.walk(root):
            if os.path.basename(dirpath) in ("archivio", "quarantena"):
                continue
            for fn in files:
                if fn.endswith((".json", ".pt")) and "truth" in fn.lower():
                    found.append(os.path.join(dirpath, fn))
    if not found:
        sys.exit("nothing found (looking for *truth*.json / *truth*.pt); "
                 "run from the studio root with no arguments")

    rows, by_hash = [], {}
    for p in sorted(found):
        ident, err = identity_from(p)
        h = sha8(p)
        ext = os.path.splitext(p)[1]
        if ident:
            canon = sanitize(
                f"truthdict_{ident['model']}_K{ident['k']}_n{ident['n']}"
                f"_seed{ident['seed']}" + ("_gauge" if ident["gauge"] else "") + ext)
        else:
            canon = f"UNKNOWN_{h}{ext}"
        rows.append(dict(path=p, canon=canon, hash=h, err=err or "",
                         created=time.strftime("%d/%m %H:%M", time.localtime(os.path.getctime(p))),
                         modified=time.strftime("%d/%m %H:%M", time.localtime(os.path.getmtime(p))),
                         **(ident or {})))
        by_hash.setdefault(h, []).append(rows[-1])

    os.makedirs("archivio", exist_ok=True)
    archived, conflicts, failures = {}, [], []
    for h, group in by_hash.items():
        r = group[0]
        if r["err"]:
            continue
        canon = r["canon"]
        if canon in archived and archived[canon] != h:
            conflicts.append((canon, archived[canon], h))
            stem, ext = os.path.splitext(canon)
            canon = f"{stem}_{h}{ext}"
        if canon not in archived:
            dst = os.path.join("archivio", canon)
            if os.path.exists(dst) and sha8(dst) != h:
                failures.append((r["path"], canon,
                                 "ARCHIVE DIVERGENCE: existing file has different "
                                 "content, NOT overwritten (resolve by hand)"))
                archived[canon] = sha8(dst)
                continue
            try:
                shutil.copy2(r["path"], dst)
                archived[canon] = h
            except OSError as e:
                failures.append((r["path"], canon, str(e)))

    print(f"{'CANONICAL NAME':52s} {'sha8':8s} {'copies':6s} created/modified (first copy)")
    for h, group in sorted(by_hash.items(), key=lambda kv: kv[1][0]["canon"]):
        r = group[0]
        tag = "  !! " + r["err"] if r["err"] else ""
        print(f"{r['canon']:52s} {h:8s} {len(group):>4}    {r['created']} / {r['modified']}{tag}")
        if len(group) > 1:
            for g in group:
                print(f"    dup: {g['path']}")
    if conflicts:
        print("\n!!! CONFLICTS (same identity, DIFFERENT content) -- resolve before any analysis:")
        for canon, h1, h2 in conflicts:
            print(f"    {canon}: {h1} vs {h2} (second archived with hash suffix)")
    else:
        print("\nno identity conflicts: every canonical name maps to exactly one content.")
    if failures:
        print("\ncopy failures (reported, not fatal):")
        for src, canon, e in failures:
            print(f"    {src} -> {canon}: {e}")

    with open("inventario.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"\n[saved] archivio/ ({len(archived)} unique files)   inventario.csv ({len(rows)} entries)")

if __name__ == "__main__":
    main()
