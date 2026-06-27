# -*- coding: utf-8 -*-
"""
canvas.py  --  PURE OBSERVATION. No classifier verdict, no gate. Paint the canvas and look.

Uses ONLY the validated part: the single-layer SVD paired-difference axis. Builds THREE
orthonormal directions from the SVD of (true - false) differences -- v1 (truth axis / Re),
v2 (Im), v3 (a genuine third dimension) -- projects every sentence, and draws:

  1) the 2D complex plane (Re, Im), true vs false, with critical line and commitment circle
  2) a real 3D scatter (v1, v2, v3) -- the triangulation
  3) per-coordinate distributions of Re, |z| and theta, so the eye SEES which coordinate
     separates true from false and which one is misleading (the 'argument' fear, tested visually)

Run it on EASY data and on HARD data and compare with your own eyes:
  python canvas.py --dataset builtin
  python canvas.py --dataset mix --max-pairs 250

It imports truth_probe.py (same folder) to reuse the validated model/data/geometry code.
Figures are saved as PNG next to the script.
"""
import os
import math
import argparse
import torch
import truth_probe as T   # reuse validated loaders, extraction, fit_axis, auc_score


def three_axes(Hl, pidx):
    """SVD of paired (true-false) differences -> v1,v2,v3 orthonormal + robust calibration.
    v1 oriented so 'true' is positive. Nothing supervised builds the axes; the pairing does."""
    Hl = Hl.float()
    it = [p[0] for p in pidx]; iff = [p[1] for p in pidx]
    D = Hl[it] - Hl[iff]
    _, _, Vh = torch.linalg.svd(D, full_matrices=False)
    v1 = Vh[0].clone()
    v2 = Vh[1] - torch.dot(Vh[1], v1) * v1; v2 = v2 / v2.norm().clamp_min(1e-8)
    v3 = Vh[2] - torch.dot(Vh[2], v1) * v1 - torch.dot(Vh[2], v2) * v2
    v3 = v3 / v3.norm().clamp_min(1e-8)
    if (Hl[it] @ v1).mean() < (Hl[iff] @ v1).mean():
        v1 = -v1
    cal = {}
    for name, v in (("1", v1), ("2", v2), ("3", v3)):
        c = Hl @ v
        med = float(c.median()); scale = float((1.4826 * (c - c.median()).abs().median()).clamp_min(1e-8))
        cal[name] = (med, scale)
    return v1, v2, v3, cal


def project(Hl, v1, v2, v3, cal):
    Hl = Hl.float()
    Re = (Hl @ v1 - cal["1"][0]) / cal["1"][1]
    Im = (Hl @ v2 - cal["2"][0]) / cal["2"][1]
    W3 = (Hl @ v3 - cal["3"][0]) / cal["3"][1]
    mod = torch.sqrt(Re ** 2 + Im ** 2)
    theta = torch.atan2(Im, Re)
    return Re, Im, W3, mod, theta


def pick_layer(H, pidx, y, layers):
    """layer with best in-sample paired-axis separation (this is a VIEWER, not a claim)."""
    best, bL = -1.0, layers[0]
    for L in layers:
        v1, _, _, cal = three_axes(H[:, L, :], pidx)
        Re = (H[:, L, :].float() @ v1 - cal["1"][0]) / cal["1"][1]
        a = T.auc_score(Re, y); a = max(a, 1 - a)
        if a > best:
            best, bL = a, L
    return bL, best


def centroid_separation(Re, Im, W3, y):
    """Quantify 'clumps vs overlapping': distance between the true/false centroids
    normalized by the average spread of the two clouds.
      >> 1  -> clumps (well separated);  ~1 or < 1 -> (overlapping).
    Computed in the 3D (v1,v2,v3) space -- it MEASURES whether zones exist,
    it does not assume them."""
    X = torch.stack([Re, Im, W3], 1)
    t = X[y == 1]; f = X[y == 0]
    ct, cf = t.mean(0), f.mean(0)
    d_centroids = float((ct - cf).norm())
    spread = 0.5 * (float(t.std(0).norm()) + float(f.std(0).norm()))
    ratio = d_centroids / max(spread, 1e-8)
    return d_centroids, spread, ratio


def draw(Re, Im, W3, mod, theta, y, r, layer, dataset, out, ratio=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    Re, Im, W3, mod, theta = (x.numpy() for x in (Re, Im, W3, mod, theta))
    y = y.numpy()
    t = y == 1; f = y == 0
    C_T, C_F = "#2c7fb8", "#d6604d"

    # ---- Fig 1: 2D complex plane (like the reference image) ----
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.scatter(Re[t], Im[t], s=28, c=C_T, alpha=0.75, label="true")
    ax.scatter(Re[f], Im[f], s=28, c=C_F, alpha=0.75, label="false")
    # centroids (the clumps): big diamonds + a line between them
    ctr_t = (Re[t].mean(), Im[t].mean()); ctr_f = (Re[f].mean(), Im[f].mean())
    ax.scatter(*ctr_t, s=320, marker="D", c=C_T, edgecolors="k", linewidths=1.5, zorder=5)
    ax.scatter(*ctr_f, s=320, marker="D", c=C_F, edgecolors="k", linewidths=1.5, zorder=5)
    ax.plot([ctr_t[0], ctr_f[0]], [ctr_t[1], ctr_f[1]], "k-", lw=1.5, zorder=4,
            label="centroid distance")
    ax.axvline(0, color="k", ls="--", lw=1.0)
    th = np.linspace(0, 2 * np.pi, 200)
    ax.plot(r * np.cos(th), r * np.sin(th), ls=":", color="gray", lw=1.0, label=f"|z|=r ({r:.2f})")
    ax.set_xlabel("Re(z)  (truth axis;  Re<0 = false side)")
    ax.set_ylabel("Im(z)  (2nd direction)")
    title = f"Canvas @layer {layer} [{dataset}]"
    if ratio is not None:
        verdict = "CLUMPS (separated)" if ratio > 1.0 else "(overlapping)"
        title += f"\ncentroid separation ratio = {ratio:.2f}  ->  {verdict}"
    ax.set_title(title)
    ax.legend(loc="best", fontsize=9); ax.grid(True, alpha=0.25)
    fig.tight_layout(); p1 = f"{out}_plane.png"; fig.savefig(p1, dpi=130); plt.close(fig)

    # ---- Fig 2: real 3D scatter (v1, v2, v3) ----
    fig = plt.figure(figsize=(8, 7)); ax = fig.add_subplot(111, projection="3d")
    ax.scatter(Re[t], Im[t], W3[t], s=22, c=C_T, alpha=0.8, label="true")
    ax.scatter(Re[f], Im[f], W3[f], s=22, c=C_F, alpha=0.8, label="false")
    ax.set_xlabel("v1 (Re, truth)"); ax.set_ylabel("v2 (Im)"); ax.set_zlabel("v3 (3rd)")
    ax.set_title(f"3D triangulation @layer {layer} [{dataset}]")
    ax.legend(fontsize=9)
    fig.tight_layout(); p2 = f"{out}_3d.png"; fig.savefig(p2, dpi=130); plt.close(fig)

    # ---- Fig 3: per-coordinate distributions -- which separates, which deceives? ----
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    for axi, (vals, name) in zip(axes, [(Re, "Re (position/truth)"),
                                        (mod, "|z| (energy/commitment)"),
                                        (theta, "theta (argument/phase)")]):
        lo, hi = float(np.min(vals)), float(np.max(vals))
        bins = np.linspace(lo, hi, 24)
        axi.hist(vals[t], bins=bins, color=C_T, alpha=0.6, label="true", density=True)
        axi.hist(vals[f], bins=bins, color=C_F, alpha=0.6, label="false", density=True)
        a = T.auc_score(torch.tensor(vals), torch.tensor(y)); a = max(a, 1 - a)
        axi.set_title(f"{name}\nseparation AUC = {a:.3f}")
        axi.legend(fontsize=8); axi.grid(True, alpha=0.2)
    fig.suptitle(f"Which coordinate separates true/false? @layer {layer} [{dataset}]", y=1.02)
    fig.tight_layout(); p3 = f"{out}_coords.png"; fig.savefig(p3, dpi=130, bbox_inches="tight"); plt.close(fig)
    return p1, p2, p3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
    ap.add_argument("--dataset", default="builtin",
                    choices=["builtin", "counterfact", "truthfulqa", "mix"])
    ap.add_argument("--max-pairs", type=int, default=250)
    ap.add_argument("--layer", type=int, default=-1, help="-1 = auto-pick best layer")
    ap.add_argument("--pool", default="last", choices=["last", "mean"])
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--dtype", default="auto", choices=["auto", "float32", "float16", "bfloat16"])
    ap.add_argument("--rev-counterfact", default=None)
    ap.add_argument("--rev-truthfulqa", default=None)
    ap.add_argument("--out", default="canvas")
    a = ap.parse_args()

    dev, dt = T.resolve_device_dtype(a.device, a.dtype)
    pairs = list(T.DEFAULT_PAIRS) if a.dataset == "builtin" else \
        T.load_pairs(a.dataset, a.max_pairs, 0, a.rev_counterfact, a.rev_truthfulqa, None, None)
    items, pidx = T.pairs_to_items(pairs)
    y = torch.tensor([l for l, _ in items])
    print(f"[data] {len(items)} sentences = {len(pidx)} pairs ({a.dataset})")
    print(f"[model] {a.model} on {dev} ({dt})")
    tok, model = T.load_model(a.model, dt, dev)
    H = T.collect(model, tok, items, dev, a.pool)
    layers = list(range(H.shape[1]))

    if a.layer < 0:
        L, auc = pick_layer(H, pidx, y, layers)
        print(f"[layer] auto-picked {L} (in-sample paired AUC {auc:.3f})")
    else:
        L = a.layer; print(f"[layer] using {L}")

    v1, v2, v3, cal = three_axes(H[:, L, :], pidx)
    Re, Im, W3, mod, theta = project(H[:, L, :], v1, v2, v3, cal)
    r = float(mod.median())

    # numeric guide for the eye: which coordinate actually separates?
    def sep(v):
        a_ = T.auc_score(v, y); return max(a_, 1 - a_)
    th_true = torch.atan2(torch.sin(theta[y == 1]).mean(), torch.cos(theta[y == 1]).mean())
    phdev = (torch.atan2(torch.sin(theta - th_true), torch.cos(theta - th_true))).abs()
    print(f"[separation AUC]  Re(position) {sep(Re):.3f} | |z|(energy) {sep(mod):.3f} "
          f"| v3(3rd) {sep(W3):.3f} | phase-dev {sep(-phdev):.3f}")
    print("  (if Re separates but |z| and phase do not, energy/argument are NOT the signal --")
    print("   the side, not the magnitude or the angle, carries truth.)")

    d_c, spread, ratio = centroid_separation(Re, Im, W3, y)
    print(f"[centroids] distance {d_c:.2f} | spread {spread:.2f} | ratio {ratio:.2f}  -> "
          + ("CLUMPS (separated)" if ratio > 1.0 else "(overlapping)"))
    print("  (ratio measures IF zones exist; it does not assume them. >1 clumps, <=1 mayo.)")

    p1, p2, p3 = draw(Re, Im, W3, mod, theta, y, r, L, a.dataset, a.out, ratio)
    print(f"\nsaved: {p1}\n       {p2}\n       {p3}")


if __name__ == "__main__":
    main()
