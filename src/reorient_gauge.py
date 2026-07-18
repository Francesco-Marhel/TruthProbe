"""
reorient_gauge.py -- re-sign per-category axes against the GLOBAL truth axis.

The within-category orientation (true side positive, decided by that
category's own margin) is an unstable gauge when the margin is thin: the
seed study showed sign storms on Qwen@60. The global truth axis has a
large, seed-stable margin; orienting every category axis by
sign(v1_c . t_global) is a declared, stable gauge. This tool loads a
dictionary bundle (.pt), reports the gauge margin |cos(v1_c, t_global)|
per category, flags thin ones, re-signs the axes, recomputes the PEAK
signed cosine matrix, and writes <name>_gauge.json next to the original
(cos_early is copied unchanged and marked: the early layer has no stable
gauge by construction). Nothing in the original bundle is modified.

Usage:  python reorient_gauge.py dizionari/truth_dictionary_Qwen25_3B.pt
        python reorient_gauge.py dizionari_seed1/*.pt --thin 0.10
"""
import argparse, glob, json, os, sys
import torch

def unit(v):
    return v / v.norm().clamp_min(1e-8)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bundles", nargs="+", help=".pt dictionary bundles (globs ok)")
    ap.add_argument("--thin", type=float, default=0.10,
                    help="margin below this = thin gauge, flagged")
    ap.add_argument("--mode", choices=["tglobal", "eigen"], default="tglobal",
                    help="reference: the bundle's t_global, or the leading "
                         "eigenvector of its own cosine matrix (sign consensus)")
    ap.add_argument("--ref", default=None,
                    help="path to another .pt bundle whose t_global is used as reference")
    a = ap.parse_args()
    paths = [p for pat in a.bundles for p in sorted(glob.glob(pat))]
    if not paths:
        sys.exit("no bundles matched")
    for path in paths:
        try:
            d = torch.load(path, map_location="cpu")
        except Exception as e:
            print(f"\n=== {path} ===\n  [skip] unreadable ({type(e).__name__})")
            continue
        missing = [k for k in ("cats", "axes", "t_global") if k not in d]
        if missing:
            print(f"\n=== {path} ===\n  [skip] missing keys: {missing}")
            continue
        cats = list(d["cats"])
        axes = d["axes"].float()              # [K, d]
        K = axes.shape[0]
        A0 = torch.stack([unit(axes[i]) for i in range(K)])
        if a.ref:
            r = torch.load(a.ref, map_location="cpu")
            tg = unit(r["t_global"].float())
            gauge_name = f"borrowed t_global from {a.ref}"
            margins_all = (A0 @ tg).tolist()
        elif a.mode == "eigen":
            C = A0 @ A0.T
            evals, evecs = torch.linalg.eigh(C)
            u = evecs[:, -1]
            if u.sum() < 0: u = -u            # convention: majority positive
            u = u / u.abs().max()             # normalize to max |component| = 1
            gauge_name = "eigen-consensus (leading eigenvector of cos matrix)"
            margins_all = u.tolist()
        else:
            tg = unit(d["t_global"].float())
            gauge_name = "global-axis (sign(v1 . t_global))"
            margins_all = (A0 @ tg).tolist()
        print(f"\n=== {path}   K={K}   gauge: {gauge_name} ===")
        signs, margins = [], []
        for i, c in enumerate(cats):
            m = float(margins_all[i])
            s = 1.0 if m >= 0 else -1.0
            flag = "  <-- THIN GAUGE (sign unreliable, report unsigned)" if abs(m) < a.thin else ""
            fl = "  [flipped]" if s < 0 else ""
            print(f"  {c:6s} margin = {m:+.3f}{fl}{flag}")
            signs.append(s); margins.append(m)
        S = torch.tensor(signs)
        axes_g = axes * S.unsqueeze(1)
        A = torch.stack([unit(axes_g[i]) for i in range(K)])
        M = (A @ A.T).tolist()
        out = {
            "model": d.get("model"), "gauge": gauge_name,
            "thin_threshold": a.thin,
            "peak_block": d.get("peak_block"), "k_relations": K,
            "pairs_per_relation": d.get("pairs_per_relation"),
            "cats": cats,
            "gauge_margins": [round(m, 4) for m in margins],
            "thin_categories": [c for c, m in zip(cats, margins) if abs(m) < a.thin],
            "flipped_vs_original": [c for c, s in zip(cats, signs) if s < 0],
            "cos_peak": [[round(float(x), 4) for x in row] for row in M],
        }
        # carry over what the stress tool needs, unchanged
        for k in ("cos_early", "transfer"):
            if k in d:
                v = d[k]
                out[k] = v.tolist() if hasattr(v, "tolist") else v
        if "cos_early" in out:
            out["cos_early_note"] = "OLD gauge (early layer has no stable gauge)"
        dst = os.path.splitext(path)[0] + "_gauge.json"
        with open(dst, "w") as f:
            json.dump(out, f, indent=1)
        print(f"  flipped: {out['flipped_vs_original'] or 'none'}   "
              f"thin: {out['thin_categories'] or 'none'}")
        print(f"  [saved] {dst}")

if __name__ == "__main__":
    main()
