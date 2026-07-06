# -*- coding: utf-8 -*-
"""
arrangement_law.py -- R5, the arrangement law, reproducibly.

Matrices transcribed from the canonical K=8 runs (identity checks 1.23e-07 /
8.37e-08; archived in v2_verify/cat8_qwen3b.txt and cat8_llama3b.txt,
July 5 2026; regenerate with:
    python categories.py --model Qwen/Qwen2.5-3B --peak 16 --write-layer 17 --k-relations 8
    python categories.py --model meta-llama/Llama-3.2-3B --peak 9 --write-layer 10 --k-relations 8
).

Statistic: Pearson/Spearman correlation between the 28 off-diagonal signed
cosines of the two PEAK matrices, against a Mantel-style category-label
permutation null (10,000 relabelings). Controls: the same for EARLY matrices
(shared lexical start), and the within-model EARLY-vs-PEAK correlation (the
reorganization). Robustness: excluding P176 (degenerate transfer on Llama).
No auto-verdict: numbers only.
"""
import random
import numpy as np

CATS = ["P103", "P1412", "P176", "P27", "P30", "P37", "P413", "P495"]

Q_PEAK = np.array([
 [1, .36, .13, .28, -.10, .27, -.02, .11],
 [.36, 1, .11, .38, -.13, .76, -.06, .09],
 [.13, .11, 1, .25, .07, .00, .00, .03],
 [.28, .38, .25, 1, -.20, .28, -.04, .16],
 [-.10, -.13, .07, -.20, 1, -.07, -.09, .05],
 [.27, .76, .00, .28, -.07, 1, -.06, .09],
 [-.02, -.06, .00, -.04, -.09, -.06, 1, -.01],
 [.11, .09, .03, .16, .05, .09, -.01, 1]])
L_PEAK = np.array([
 [1, .69, .40, .66, -.14, .56, .00, .51],
 [.69, 1, .38, .62, -.15, .62, .01, .52],
 [.40, .38, 1, .51, -.01, .40, .06, .51],
 [.66, .62, .51, 1, -.09, .55, .06, .75],
 [-.14, -.15, -.01, -.09, 1, -.09, -.06, -.03],
 [.56, .62, .40, .55, -.09, 1, .03, .52],
 [.00, .01, .06, .06, -.06, .03, 1, .06],
 [.51, .52, .51, .75, -.03, .52, .06, 1]])
Q_EARLY = np.array([
 [1, .31, .04, -.19, -.08, -.26, -.05, .09],
 [.31, 1, -.02, -.24, -.03, -.90, -.09, .08],
 [.04, -.02, 1, -.03, .01, .03, .03, .01],
 [-.19, -.24, -.03, 1, .23, .23, .03, .27],
 [-.08, -.03, .01, .23, 1, .04, -.10, .13],
 [-.26, -.90, .03, .23, .04, 1, .08, -.09],
 [-.05, -.09, .03, .03, -.10, .08, 1, .01],
 [.09, .08, .01, .27, .13, -.09, .01, 1]])
L_EARLY = np.array([
 [1, .08, .05, -.05, -.06, -.18, -.03, .03],
 [.08, 1, .04, -.16, -.06, -.86, -.03, .08],
 [.05, .04, 1, .10, -.05, -.04, -.00, .08],
 [-.05, -.16, .10, 1, .17, .21, .04, .11],
 [-.06, -.06, -.05, .17, 1, .08, -.01, .07],
 [-.18, -.86, -.04, .21, .08, 1, .03, -.11],
 [-.03, -.03, -.00, .04, -.01, .03, 1, -.01],
 [.03, .08, .08, .11, .07, -.11, -.01, 1]])


def offdiag(M, idx=None):
    idx = idx if idx is not None else list(range(M.shape[0]))
    return np.array([M[i, j] for a, i in enumerate(idx) for j in idx[a + 1:]])


def pearson(a, b):
    return float(np.corrcoef(a, b)[0, 1])


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    return pearson(ra, rb)


def mantel(A, B, idx=None, perms=10000, seed=0):
    idx = idx if idx is not None else list(range(A.shape[0]))
    a = offdiag(A, idx)
    r_obs = pearson(a, offdiag(B, idx))
    rng = random.Random(seed)
    null = []
    for _ in range(perms):
        pi = list(idx)
        rng.shuffle(pi)
        null.append(pearson(a, offdiag(B[np.ix_(pi, pi)])))
    null = np.array(null)
    p = (1 + int((null >= r_obs).sum())) / (perms + 1)
    return r_obs, float(null.mean()), float(np.quantile(null, 0.95)), p


if __name__ == "__main__":
    print("=== R5: arrangement law (PEAK vs PEAK, 28 off-diagonal cells) ===")
    r, nm, n95, p = mantel(Q_PEAK, L_PEAK)
    print(f"  Pearson r = {r:+.3f}   Spearman = "
          f"{spearman(offdiag(Q_PEAK), offdiag(L_PEAK)):+.3f}")
    print(f"  Mantel null: mean {nm:+.3f}  95pct {n95:+.3f}   p = {p:.4f}")
    keep = [0, 1, 3, 4, 5, 6, 7]
    r2, _, n952, p2 = mantel(Q_PEAK, L_PEAK, idx=keep)
    print(f"  robustness (P176 excluded, 21 cells): r = {r2:+.3f}  "
          f"null 95pct {n952:+.3f}  p = {p2:.4f}")
    print("\n=== controls ===")
    re, _, _, pe = mantel(Q_EARLY, L_EARLY)
    print(f"  EARLY vs EARLY (shared lexical start): r = {re:+.3f}  p = {pe:.4f}")
    print(f"  within-model EARLY vs PEAK:  Qwen r = "
          f"{pearson(offdiag(Q_EARLY), offdiag(Q_PEAK)):+.3f}   Llama r = "
          f"{pearson(offdiag(L_EARLY), offdiag(L_PEAK)):+.3f}")
    print("  reading: shared start (high early-early), shared destination")
    print("  (high peak-peak), reached by a reorganization (low/negative")
    print("  within-model early-peak) that both families execute alike.")
