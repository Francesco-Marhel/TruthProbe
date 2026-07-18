"""
arrangement_stress_test.py -- does the arrangement law survive at scale?

The registered law (R5) is CROSS-FAMILY AGREEMENT at matched protocol,
not stability of one family's matrix across sample sizes. This tool
compares two dictionaries (one per family, same pairs-per-relation) and
prints the Mantel test on the signed peak cosines, the early-block
surface control, and a restricted variant on the categories whose
within-AUC clears a pre-declared threshold on BOTH models. It also
reports, against optional reference dictionaries (the 60-pair ones),
which categories look sign-flipped. No GPU, no model: pure arithmetic
on files already produced.

Examples (from the folder with the JSONs):
  python arrangement_stress_test.py --a dizionari/qwen_888.json --b dizionari/llama_888.json \
      --ref-a dizionari/qwen_60.json --ref-b dizionari/llama_60.json --auc-threshold 0.6
  python arrangement_stress_test.py --a qwen_888.json --b llama_888.json --exclude P176
"""
import argparse, json, math, os, random

def load_dict(path):
    if path.endswith(".pt") or path.endswith(".pts"):
        import torch
        d = torch.load(path, map_location="cpu")
        out = {}
        for k in ("cats", "cos_peak", "cos_early", "transfer", "model"):
            v = d.get(k)
            out[k] = v.tolist() if hasattr(v, "tolist") else v
        return out
    with open(path) as f:
        return json.load(f)

def offdiag(M, idx):
    return [M[i][j] for i in idx for j in idx if i < j]

def pearson(x, y):
    n = len(x); mx = sum(x)/n; my = sum(y)/n
    sx = math.sqrt(sum((a-mx)**2 for a in x)); sy = math.sqrt(sum((b-my)**2 for b in y))
    if sx == 0 or sy == 0: return float("nan")
    return sum((a-mx)*(b-my) for a, b in zip(x, y))/(sx*sy)

def mantel(Ma, Mb, idx, perms=9999, seed=0):
    x = offdiag(Ma, idx); y = offdiag(Mb, idx)
    r_obs = pearson(x, y)
    rng = random.Random(seed); ge = 1
    for _ in range(perms):
        p = list(idx); rng.shuffle(p)
        yp = [Mb[i][j] if i < j else Mb[j][i] for a, i in enumerate(p) for b, j in enumerate(p) if a < b]
        if pearson(x, yp) >= r_obs: ge += 1
    return r_obs, ge/(perms+1)

def spearman(x, y):
    def ranks(v):
        s = sorted(range(len(v)), key=lambda i: v[i]); r = [0]*len(v)
        for k, i in enumerate(s): r[i] = k
        return r
    return pearson(ranks(x), ranks(y))

def best_sign_flips(M_new, M_ref, cats):
    """Greedy: flip category signs in M_new to maximize agreement with M_ref.
    Reports which flips help -> suspected orientation flips."""
    K = len(cats); signs = [1]*K
    def corr(sg):
        x = [sg[i]*sg[j]*M_new[i][j] for i in range(K) for j in range(K) if i < j]
        y = [M_ref[i][j] for i in range(K) for j in range(K) if i < j]
        return pearson(x, y)
    improved = True
    while improved:
        improved = False
        for i in range(K):
            base = corr(signs); signs[i] *= -1
            if corr(signs) > base + 1e-9: improved = True
            else: signs[i] *= -1
    return [c for c, s in zip(cats, signs) if s < 0], corr(signs)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="dictionary of family A (e.g. Qwen @888)")
    ap.add_argument("--b", required=True, help="dictionary of family B (e.g. Llama @888)")
    ap.add_argument("--ref-a", default=None, help="reference for A (e.g. Qwen @60)")
    ap.add_argument("--ref-b", default=None, help="reference for B (e.g. Llama @60)")
    ap.add_argument("--auc-threshold", type=float, default=0.6)
    ap.add_argument("--exclude", nargs="*", default=[])
    ap.add_argument("--perms", type=int, default=9999)
    a, b = None, None
    args = ap.parse_args()
    A, B = load_dict(args.a), load_dict(args.b)
    assert A["cats"] == B["cats"], "category sets differ"
    cats = A["cats"]; K = len(cats)
    print(f"[A] {A.get('model','?')}   [B] {B.get('model','?')}   K={K}")

    idx_all = [i for i, c in enumerate(cats) if c not in args.exclude]
    if args.exclude: print(f"[excluded by flag] {args.exclude}")

    r, p = mantel(A["cos_peak"], B["cos_peak"], idx_all, args.perms)
    x, y = offdiag(A["cos_peak"], idx_all), offdiag(B["cos_peak"], idx_all)
    print(f"\nTHE LAW @ this protocol   Mantel r = {r:+.3f}   p = {p:.4f}   "
          f"Spearman = {spearman(x, y):+.3f}   (K'={len(idx_all)})")
    re, pe = mantel(A["cos_early"], B["cos_early"], idx_all, args.perms)
    print(f"surface control (early)   Mantel r = {re:+.3f}   p = {pe:.4f}")

    diagA = [A["transfer"][i][i] for i in range(K)]
    diagB = [B["transfer"][i][i] for i in range(K)]
    keep = [i for i in idx_all if diagA[i] >= args.auc_threshold and diagB[i] >= args.auc_threshold]
    print(f"\nwithin-AUC (A|B): " + "  ".join(f"{cats[i]} {diagA[i]:.2f}|{diagB[i]:.2f}" for i in range(K)))
    print(f"restricted set (within-AUC >= {args.auc_threshold} on BOTH): {[cats[i] for i in keep]}")
    if len(keep) >= 4:
        rr, rp = mantel(A["cos_peak"], B["cos_peak"], keep, args.perms)
        print(f"THE LAW, restricted       Mantel r = {rr:+.3f}   p = {rp:.4f}   (K'={len(keep)}; "
              f"note: with K'<6 the permutation floor limits p)")
    else:
        print("restricted set too small for a meaningful Mantel (need >= 4)")

    # --- symbiosis: per-category adherence vs knowledge proxy ---
    def row_adherence(i):
        x = [A["cos_peak"][i][j] for j in idx_all if j != i]
        y = [B["cos_peak"][i][j] for j in idx_all if j != i]
        return pearson(x, y)
    adh = {cats[i]: row_adherence(i) for i in idx_all}
    know = {cats[i]: min(diagA[i], diagB[i]) for i in idx_all}
    order = sorted(adh, key=lambda c: -adh[c])
    print("\nSYMBIOSIS  per-category adherence corr(row_A, row_B) vs knowledge proxy min(within-AUC):")
    for c in order:
        print(f"  {c:6s} adherence {adh[c]:+.3f}   knowledge {know[c]:.2f}")
    ks = [know[c] for c in order]; av = [adh[c] for c in order]
    print(f"  rank-correlation (Spearman) adherence ~ knowledge: {spearman(av, ks):+.3f}"
          f"   Pearson: {pearson(av, ks):+.3f}   (registered prediction: positive)")

    for tag, new, ref in (("A", args.a, args.ref_a), ("B", args.b, args.ref_b)):
        if not ref: continue
        R = load_dict(ref); assert R["cats"] == cats
        M_new = (A if tag == "A" else B)["cos_peak"]
        flips, c_after = best_sign_flips(M_new, R["cos_peak"], cats)
        c_before = pearson(offdiag(M_new, range(K)), offdiag(R["cos_peak"], range(K)))
        print(f"\n[{tag} vs its reference] corr before sign repair {c_before:+.3f} -> after {c_after:+.3f}"
              f"   suspected orientation flips: {flips if flips else 'none'}")

if __name__ == "__main__":
    main()
