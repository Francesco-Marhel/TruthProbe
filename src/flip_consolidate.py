# -*- coding: utf-8 -*-
"""
flip_consolidate.py  --  Consolidate the peak-relative FFN flip on ONE extraction.

Three shields in one run (extraction is the only expensive step; everything
below is CPU loops on cached tensors):

  (a) SEEDS      -- the per-layer gaps of attn/ffn contributions on the fixed
                    truth axis, mean+-std over N cross-validation seeds. A gap
                    is 'stable' only if |mean| > max(2*std, 0.03), as in
                    anatomy_consolidate.
  (b) PERM NULL  -- at the flip layer, swap true/false within pairs and refit:
                    the null distribution of the gap. Two-sided p on |gap|.
  (c) ROTATION   -- the objection raised in-session: is the negative gap a
                    rotating write direction seen through a fixed axis (a
                    'cosine sweep'), rather than genuine anti-truth writing?
                    Per layer we take the mean intra-pair FFN difference vector
                    and split it into along-axis and orthogonal components.
                      rotation predicts : ~constant norm, cosine sweeping
                                          smoothly through zero, orthogonal
                                          large where along-axis is small
                      erosion  predicts : abrupt sign jump at peak+1 with the
                                          along-axis magnitude intact
                    (Descriptive, full-fit axis: labeled Obs.)

Defaults target the 3B run: axis block 16 (= truth peak), flip layer 17,
scan 12-23. Reuses truth_probe.py + anatomy.py. fp32 (identity check gates it).

    python flip_consolidate.py --model Qwen/Qwen2.5-3B
"""
import argparse
import random
import torch
import truth_probe as T
import anatomy as A


def axis_per_fold(Hax, pidx, folds, seed, orient=None):
    """Fit the fixed axis once per fold (reused across all layers)."""
    axes = []
    for tr, te in T.kfold_pairs(len(pidx), folds, seed):
        o = None if orient is None else [orient[p] for p in tr]
        axes.append((T.fit_axis(Hax, [pidx[p] for p in tr], o), te))
    return axes


def gap_on_axis(Hc, pidx, axes, orient=None):
    """Held-out class gap (and d') of a contribution on prefit fold axes."""
    gaps, ds = [], []
    for ax, te in axes:
        pt, pf = [], []
        for p in te:
            it, iff = pidx[p]
            if orient is None or orient[p] >= 0:
                pt.append(it); pf.append(iff)
            else:
                pt.append(iff); pf.append(it)
        prt = Hc[pt].float() @ ax["v1"]; prf = Hc[pf].float() @ ax["v1"]
        gap = float(prt.mean() - prf.mean())
        pooled = float(torch.sqrt((prt.var() + prf.var()) / 2).clamp_min(1e-8))
        gaps.append(gap); ds.append(gap / pooled)
    return sum(gaps) / len(gaps), sum(ds) / len(ds)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B")
    ap.add_argument("--dataset", default="counterfact",
                    choices=["builtin", "counterfact", "truthfulqa", "mix"])
    ap.add_argument("--max-pairs", type=int, default=250)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--perm", type=int, default=100)
    ap.add_argument("--axis-block", type=int, default=16, help="truth-peak block (v1 source)")
    ap.add_argument("--flip-layer", type=int, default=17, help="layer for the permutation null")
    ap.add_argument("--scan-start", type=int, default=12)
    ap.add_argument("--scan-end", type=int, default=23)
    ap.add_argument("--rev-counterfact", default="c945b082ca08d0a8f3ba227fb78404a09614c36e")
    ap.add_argument("--rev-truthfulqa", default="741b8276f2d1982aa3d5b832d3ee81ed3b896490")
    ap.add_argument("--file-counterfact", default=None)
    ap.add_argument("--file-truthfulqa", default=None)
    a = ap.parse_args()

    dev, _ = T.resolve_device_dtype("auto", "auto")
    dt = torch.float32
    print("[task flip_consolidate] seeds + permutation null + rotation check, one extraction")
    pairs = list(T.DEFAULT_PAIRS) if a.dataset == "builtin" else \
        T.load_pairs(a.dataset, a.max_pairs, 0, a.rev_counterfact,
                     a.rev_truthfulqa, a.file_counterfact, a.file_truthfulqa)
    items, pidx = T.pairs_to_items(pairs)
    print(f"[data] {len(items)} sentences = {len(pidx)} pairs")
    print(f"[model] {a.model} on {dev} ({dt})")
    tok, model = T.load_model(a.model, dt, dev)

    H_resid, H_attn, H_ffn = A.collect_components(model, tok, items, dev)
    del model
    rel = A.identity_check(H_resid, H_attn, H_ffn)
    print(f"[identity check] median {float(rel.median()):.2e}  (must be ~0)")
    if float(rel.median()) > 1e-3:
        print("  ABORT: decomposition invalid."); return

    L = H_attn.shape[1]
    scan = list(range(a.scan_start, min(a.scan_end, L - 1) + 1))
    Hax = H_resid[:, a.axis_block + 1, :]

    # ---------- (a) seeds ----------
    print(f"\n=== (a) stability over {a.seeds} seeds, fixed axis @block {a.axis_block} ===")
    fold_axes = [axis_per_fold(Hax, pidx, a.folds, s) for s in range(a.seeds)]
    print(f"{'layer':>5} | {'gap_attn':>16} | {'gap_ffn':>16} | {'d\'_ffn':>13} | verdict")
    print("-" * 75)
    for Lb in scan:
        ga, gf, df = [], [], []
        for s in range(a.seeds):
            g1, _ = gap_on_axis(H_attn[:, Lb, :], pidx, fold_axes[s])
            g2, d2 = gap_on_axis(H_ffn[:, Lb, :], pidx, fold_axes[s])
            ga.append(g1); gf.append(g2); df.append(d2)
        gam, gas = float(torch.tensor(ga).mean()), float(torch.tensor(ga).std())
        gfm, gfs = float(torch.tensor(gf).mean()), float(torch.tensor(gf).std())
        dfm = float(torch.tensor(df).mean())
        stable = abs(gfm) > max(2 * gfs, 0.03)
        tag = ("ANTI-truth STABLE" if gfm < 0 else "pro-truth STABLE") if stable else "~ not stable"
        mark = "<- flip" if Lb == a.flip_layer else ""
        print(f"{Lb:>5} | {gam:>+7.3f}±{gas:<6.3f} | {gfm:>+7.3f}±{gfs:<6.3f} | "
              f"{dfm:>+13.2f} | {tag} {mark}")

    # ---------- (b) permutation null at the flip layer ----------
    print(f"\n=== (b) permutation null, gap_ffn @layer {a.flip_layer} ({a.perm} reps) ===")
    obs = float(torch.tensor(
        [gap_on_axis(H_ffn[:, a.flip_layer, :], pidx, fold_axes[s])[0]
         for s in range(a.seeds)]).mean())
    rng = random.Random(0); null = []
    Hf = H_ffn[:, a.flip_layer, :]
    for b in range(a.perm):
        o = [1 if rng.random() < 0.5 else -1 for _ in pidx]
        axes = axis_per_fold(Hax, pidx, a.folds, 1000 + b, orient=o)
        g, _ = gap_on_axis(Hf, pidx, axes, orient=o)
        null.append(g)
        print(f"\r[permutation] {b+1}/{a.perm}", end="", flush=True)
    print()
    nt = torch.tensor(null)
    p = (1 + int((nt.abs() >= abs(obs)).sum())) / (a.perm + 1)
    print(f"  observed {obs:+.3f}   null mean {float(nt.mean()):+.3f}  "
          f"null |95pct| {float(nt.abs().quantile(0.95)):.3f}   p={p:.4f} (two-sided)")
    print("  -> " + ("the flip gap is REAL beyond within-pair label noise."
                     if p < 0.05 else "NOT beyond the null: do not build on this layer."))

    # ---------- (c) rotation vs erosion (descriptive, Obs.) ----------
    print(f"\n=== (c) rotation vs erosion: mean FFN pair-difference vs fixed v1 (Obs.) ===")
    ax_full = T.fit_axis(Hax, pidx)
    v1 = ax_full["v1"]
    print(f"{'layer':>5} | {'||d||':>7} | {'along':>8} | {'orth':>7} | {'cos(d,v1)':>9}")
    print("-" * 50)
    coss = {}
    for Lb in scan:
        t_idx = [it for it, _ in pidx]; f_idx = [iff for _, iff in pidx]
        d = (H_ffn[t_idx, Lb, :].float() - H_ffn[f_idx, Lb, :].float()).mean(0)
        nrm = float(d.norm())
        along = float(d @ v1)
        orth = (nrm ** 2 - along ** 2) ** 0.5
        cos = along / max(nrm, 1e-8)
        coss[Lb] = cos
        mark = "<- flip" if Lb == a.flip_layer else ""
        print(f"{Lb:>5} | {nrm:>7.2f} | {along:>+8.3f} | {orth:>7.2f} | {cos:>+9.3f} {mark}")
    pk, fl = a.axis_block, a.flip_layer
    if pk in coss and fl in coss:
        jump = coss[fl] - coss[pk]
        print(f"\n  cos jump peak->flip: {coss[pk]:+.3f} -> {coss[fl]:+.3f}  (delta {jump:+.3f})")
        print("  rotation predicts a SMOOTH sweep of cos through zero across several layers")
        print("  with ~constant ||d||; erosion predicts an abrupt sign change at peak+1.")
        print("  Read the column: if the sign flips in ONE step while ||d|| stays of the")
        print("  same order, the sign-artifact/rotation objection is CLOSED.")


if __name__ == "__main__":
    main()
