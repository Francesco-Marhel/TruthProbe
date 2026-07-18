# -*- coding: utf-8 -*-
"""
visualizza_dizionario.py  --  Read a saved truth dictionary and turn it into
readable numbers + heatmaps, plus an OPTIONAL depth *timelapse* (how the
per-category geometry evolves across layers).

Format: prefer the **.pt** bundle -- it is full float32 precision and the only
format that carries the per-layer stack `cos_by_layer` (produced by
`crea_dizionario.py --all-layers`), which the timelapse needs. The **.json** is
accepted for the three static matrices only (3-decimal, no timelapse).

Colours: a diverging map with a NEUTRAL midpoint -- signed cosines centred at 0,
transfer AUC centred at 0.5 (chance). No rainbow. The diagonal of a cosine matrix
is 1 by construction (unit axes), so all the signal is off-diagonal.

    python visualizza_dizionario.py                                   # newest .pt in dizionari/
    python visualizza_dizionario.py dizionari/truth_dictionary_Qwen25_3B.pt
    python visualizza_dizionario.py <bundle.pt> --timelapse           # + evolution curve + GIF
    python visualizza_dizionario.py <bundle.pt> --flip                # + FFN pro->anti flip curve
    python visualizza_dizionario.py <bundle> --out-dir grafici --fps 3

Outputs (PNG heatmaps; PNG evolution curve + GIF for the timelapse) go to --out-dir
(default: grafici/). Nothing existing is overwritten except files with the same name.
"""
import argparse
import glob
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")                        # headless: save files, no display needed
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.animation import FuncAnimation, PillowWriter

CMAP = "coolwarm"     # diverging, light neutral midpoint, reasonably CVD-safe


# =====================================================================
#  load (accept .pt full-precision, or .json for the static matrices)
# =====================================================================
def newest_bundle(folder="dizionari"):
    cands = sorted(glob.glob(os.path.join(folder, "*.pt")), key=os.path.getmtime, reverse=True)
    if not cands:
        raise SystemExit(f"[error] no .pt bundle in {folder}/. Pass a path, or run "
                         "crea_dizionario.py first.")
    return cands[0]


def load_bundle(path):
    if path.lower().endswith(".json"):
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        meta = {k: v for k, v in d.items() if k not in ("cos_peak", "cos_early", "transfer")}
        arr = lambda k: (np.asarray(d[k], float) if d.get(k) is not None else None)
        return dict(cats=list(d["cats"]), meta=meta, cos_peak=arr("cos_peak"),
                    cos_early=arr("cos_early"), transfer=arr("transfer"),
                    cos_by_layer=None, flip=None)
    import torch                             # only needed for the .pt path
    d = torch.load(path, weights_only=False)
    npy = lambda x: (x.detach().cpu().numpy() if hasattr(x, "detach") else
                     (np.asarray(x) if x is not None else None))
    return dict(cats=list(d["cats"]), meta=d.get("meta", {}),
                cos_peak=npy(d.get("cos_peak")), cos_early=npy(d.get("cos_early")),
                transfer=npy(d.get("transfer")), cos_by_layer=npy(d.get("cos_by_layer")),
                flip=d.get("flip"))          # dict of python lists (per-block gap/d')


# =====================================================================
#  readable numeric summary
# =====================================================================
def offvals(M, symmetric=True):
    K = M.shape[0]
    if symmetric:                            # cosine matrices are symmetric -> upper triangle
        return M[np.triu_indices(K, k=1)]
    return M[~np.eye(K, dtype=bool)]         # transfer is NOT symmetric -> full off-diagonal


def print_summary(b):
    cats, meta = b["cats"], b["meta"]
    K = len(cats)
    print("=" * 82)
    print(f"DICTIONARY  {meta.get('model', '?')}    K={K}    peak block {meta.get('peak_block', '?')}")
    print("=" * 82)
    op = offvals(b["cos_peak"])
    print(f"cos @ PEAK   off-diagonal ({len(op)} unique pairs):")
    print(f"   mean {op.mean():+.3f}   std {op.std():.3f}   min {op.min():+.3f}   "
          f"max {op.max():+.3f}   neg {100*(op < 0).mean():.0f}%   mean|cos| {np.abs(op).mean():.3f}")
    if b["cos_early"] is not None:
        oe = offvals(b["cos_early"])
        emerges = abs(op.mean()) > abs(oe.mean()) + 0.03
        print(f"cos @ EARLY  (surface control): mean {oe.mean():+.3f}   std {oe.std():.3f}   "
              f"mean|cos| {np.abs(oe).mean():.3f}")
        print(f"   -> shared-core mean  PEAK {op.mean():+.3f}  vs  EARLY {oe.mean():+.3f}   "
              f"({'structure EMERGES at the peak' if emerges else 'similar -> possibly lexical'})")
    if b["transfer"] is not None:
        T = b["transfer"]; diag = np.diag(T); off = offvals(T, symmetric=False)
        print(f"transfer AUC: within (diagonal) {diag.mean():.3f}   cross (off-diag) {off.mean():.3f}"
              f"   gap {diag.mean() - off.mean():+.3f}   (within > cross = category-specific)")
    dec = meta.get("decoding")
    if dec:
        print(f"decoding: Delta f {dec['delta_f']['acc']:.1%}  vs lexical {dec['lexical']['acc']:.1%}"
              f"   (chance {1/K:.1%}, p={dec['delta_f']['p']:.4f})")
    fl = b.get("flip")
    if fl:
        pb, fb, dpf = fl.get("axis_block"), fl.get("flip_block"), fl["dprime_ffn"]
        if pb is not None and fb is not None and fb < len(dpf):
            tag = "anti-truth FLIP" if dpf[fb] < 0 else "still pro"
            print(f"FFN flip: d'_ffn @peak {pb} = {dpf[pb]:+.2f} (pro-truth)  ->  "
                  f"@block {fb} = {dpf[fb]:+.2f} ({tag})")

    P, T = b["cos_peak"], b["transfer"]
    def keyf(i):
        return T[i, i] if T is not None else np.abs(np.delete(P[i], i)).mean()
    print("\nper-category (sorted by within-AUC if present, else by mean|cos|):")
    for i in sorted(range(K), key=keyf, reverse=True):
        rv = P[i].copy(); rv[i] = -np.inf; ja = int(np.argmax(rv))
        rz = P[i].copy(); rz[i] = np.inf; jz = int(np.argmin(rz))
        within = f"{T[i, i]:.2f}" if T is not None else "  -"
        mabs = np.abs(np.delete(P[i], i)).mean()
        print(f"  {cats[i]:<7} within-AUC {within}   mean|cos| {mabs:.2f}   "
              f"most-aligned {cats[ja]} ({P[i, ja]:+.2f})   most-anti {cats[jz]} ({P[i, jz]:+.2f})")


# =====================================================================
#  plots
# =====================================================================
def _norm(kind):
    return TwoSlopeNorm(vcenter=0.5, vmin=0.0, vmax=1.0) if kind == "auc" \
        else TwoSlopeNorm(vcenter=0.0, vmin=-1.0, vmax=1.0)


def _label_axes(ax, cats):
    K = len(cats); fs = 8 if K <= 16 else 6
    ax.set_xticks(range(K)); ax.set_yticks(range(K))
    ax.set_xticklabels(cats, rotation=90, fontsize=fs); ax.set_yticklabels(cats, fontsize=fs)
    ax.set_xticks(np.arange(-.5, K, 1), minor=True); ax.set_yticks(np.arange(-.5, K, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.5); ax.tick_params(which="minor", length=0)


def heatmap(M, cats, title, out_png, kind="cos"):
    K = len(cats)
    fig, ax = plt.subplots(figsize=(max(6, K * 0.32 + 2), max(5, K * 0.32 + 1.6)))
    im = ax.imshow(M, cmap=CMAP, norm=_norm(kind))
    _label_axes(ax, cats)
    ax.set_title(title, fontsize=11)
    if K <= 12:                              # annotate only when it stays readable
        for i in range(K):
            for j in range(K):
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=6)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("signed cosine" if kind == "cos" else "held-out AUC", fontsize=9)
    fig.tight_layout(); fig.savefig(out_png, dpi=140); plt.close(fig)
    print(f"[saved] {out_png}")


def evolution_curve(cos_by_layer, meta, out_png):
    nH = cos_by_layer.shape[0]; iu = np.triu_indices(cos_by_layer.shape[1], k=1)
    off_mean = np.array([cos_by_layer[h][iu].mean() for h in range(nH)])
    off_abs = np.array([np.abs(cos_by_layer[h][iu]).mean() for h in range(nH)])
    peak_h = meta.get("peak_hidden", (meta.get("peak_block") + 1) if meta.get("peak_block") is not None else None)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(range(nH), off_mean, "-o", ms=4, lw=2, label="off-diag mean (shared core)")
    ax.plot(range(nH), off_abs, "-s", ms=4, lw=2, label="off-diag mean |cos| (structure)")
    ax.axhline(0, color="gray", lw=1)
    if peak_h is not None and 0 <= peak_h < nH:
        ax.axvline(peak_h, color="crimson", ls="--", lw=1.5, label=f"truth peak (hidden {peak_h})")
    ax.set_xlabel("hidden layer"); ax.set_ylabel("cosine")
    ax.set_title("Category-geometry evolution across depth")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out_png, dpi=140); plt.close(fig)
    print(f"[saved] {out_png}")


def timelapse(cos_by_layer, cats, meta, out_gif, fps=3):
    nH, K, _ = cos_by_layer.shape
    peak_h = meta.get("peak_hidden", (meta.get("peak_block") + 1) if meta.get("peak_block") is not None else -1)
    iu = np.triu_indices(K, k=1)
    fig, ax = plt.subplots(figsize=(max(6, K * 0.32 + 2), max(5, K * 0.32 + 1.8)))
    im = ax.imshow(cos_by_layer[0], cmap=CMAP, norm=_norm("cos"))
    _label_axes(ax, cats)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04); cb.set_label("signed cosine", fontsize=9)
    ttl = ax.set_title("", fontsize=11)

    def frame(h):
        im.set_data(cos_by_layer[h])
        off = cos_by_layer[h][iu]
        mark = "   <== TRUTH PEAK" if h == peak_h else ""
        ttl.set_text(f"hidden layer {h}/{nH-1}    off-diag mean {off.mean():+.3f}{mark}")
        return [im, ttl]

    anim = FuncAnimation(fig, frame, frames=nH, interval=1000 / max(fps, 1), blit=False)
    anim.save(out_gif, writer=PillowWriter(fps=fps)); plt.close(fig)
    print(f"[saved] {out_gif}")


def flip_plot(flip, out_png):
    """Line chart of the FFN (and attention) contribution gap d' on the fixed truth
    axis, per block: pro-truth (>0) up to the peak, ANTI-truth (<0) from peak+1."""
    dpf, dpa = flip["dprime_ffn"], flip["dprime_attn"]
    nB = len(dpf); x = list(range(nB))
    axb, flb = flip.get("axis_block"), flip.get("flip_block")
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.plot(x, dpf, "-o", ms=4, lw=2.2, color="#c0392b", label="FFN d' (on the truth axis)")
    ax.plot(x, dpa, "-s", ms=3.5, lw=1.6, color="#2c7fb8", label="attention d'")
    ax.axhline(0, color="gray", lw=1)
    if axb is not None:
        ax.axvline(axb, color="seagreen", ls="--", lw=1.3, label=f"truth peak / axis block {axb}")
    if flb is not None and flb < nB:
        ax.axvline(flb, color="crimson", ls=":", lw=1.8, label=f"flip block {flb} (peak+1)")
        ax.annotate(f"{dpf[flb]:+.2f}", (flb, dpf[flb]), textcoords="offset points",
                    xytext=(6, -12), color="#c0392b", fontsize=9)
    ax.set_xlabel("block"); ax.set_ylabel("class gap d'   (pro-truth > 0,  anti-truth < 0)")
    ax.set_title("FFN contribution on the fixed truth axis — the pro→anti-truth flip")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out_png, dpi=140); plt.close(fig)
    print(f"[saved] {out_png}")


# =====================================================================
#  CLI
# =====================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bundle", nargs="?", default=None,
                    help=".pt (preferred) or .json; default: newest .pt in dizionari/")
    ap.add_argument("--out-dir", default="grafici", help="folder for the images (default: grafici)")
    ap.add_argument("--timelapse", action="store_true",
                    help="also render the per-layer evolution curve + animated GIF "
                         "(needs a .pt built with crea_dizionario.py --all-layers)")
    ap.add_argument("--flip", action="store_true",
                    help="also render the FFN pro->anti-truth flip curve "
                         "(needs a .pt built with crea_dizionario.py --flip-layers)")
    ap.add_argument("--fps", type=int, default=3, help="frames per second of the timelapse GIF")
    ap.add_argument("--no-summary", action="store_true", help="skip the printed numeric summary")
    a = ap.parse_args()

    path = a.bundle or newest_bundle()
    print(f"[bundle] {path}")
    b = load_bundle(path)
    os.makedirs(a.out_dir, exist_ok=True)
    tag = os.path.splitext(os.path.basename(path))[0]

    if not a.no_summary:
        print_summary(b)

    heatmap(b["cos_peak"], b["cats"], f"{tag}  -  cos @ peak",
            os.path.join(a.out_dir, f"{tag}_cos_peak.png"), "cos")
    if b["cos_early"] is not None:
        heatmap(b["cos_early"], b["cats"], f"{tag}  -  cos @ early (surface control)",
                os.path.join(a.out_dir, f"{tag}_cos_early.png"), "cos")
    if b["transfer"] is not None:
        heatmap(b["transfer"], b["cats"], f"{tag}  -  transfer AUC (within vs cross)",
                os.path.join(a.out_dir, f"{tag}_transfer.png"), "auc")

    if a.timelapse:
        if b["cos_by_layer"] is None:
            print("[timelapse] this bundle has no 'cos_by_layer'. Re-create it with "
                  "`crea_dizionario.py --all-layers` and pass the .pt (not the .json).")
        else:
            evolution_curve(b["cos_by_layer"], b["meta"],
                            os.path.join(a.out_dir, f"{tag}_evolution.png"))
            timelapse(b["cos_by_layer"], b["cats"], b["meta"],
                      os.path.join(a.out_dir, f"{tag}_timelapse.gif"), a.fps)

    if a.flip:
        if b.get("flip") is None:
            print("[flip] this bundle has no 'flip'. Re-create it with "
                  "`crea_dizionario.py --flip-layers` and pass the .pt (not the .json).")
        else:
            flip_plot(b["flip"], os.path.join(a.out_dir, f"{tag}_flip.png"))

    print(f"\n[done] images in ./{a.out_dir}/")


if __name__ == "__main__":
    main()
