# -*- coding: utf-8 -*-
"""
ffn_erosion.py  --  WHY does removing the FFN in the transition band RAISE truth AUC?

The anatomy note found (CounterFact, band 16-18, readout block 18):
    intact 0.712   FFN-off 0.815 (drop -0.103)   attn-off 0.708 (~nothing)
Two competing stories:

  H1 (active erosion): the FFN in 16-18 WRITES content (format / next-token shift)
     that lowers true/false separability -- it pushes states along or across the
     truth axis in a class-damaging way.

  H2 (freeze confound): zeroing the FFN in 16-18 simply FREEZES the stream near
     the peak (block 15). Any component that writes a lot after the peak would
     look 'erosive'; the FFN is just the dominant writer there. Nothing special.

This script separates them with four isolated subcommands (exploration; held-out
CV over pairs everywhere a classifier-style number is reported; no permutation
nulls yet -- add them once we know which numbers matter):

  python ffn_erosion.py contrib   # one pass: norms + class-gap of a_l, f_l on a FIXED axis
  python ffn_erosion.py ablate    # 3 passes: refit vs FIXED-axis AUC + FROZEN control
  python ffn_erosion.py sweep     # FFN-off one layer at a time + cumulative
  python ffn_erosion.py lens      # qualitative logit-lens of the FFN contributions

Key controls built in:
  * FROZEN control: with BOTH attn and FFN zeroed over [band], the state at the
    readout is IDENTICALLY the state entering the band, so the frozen number is
    read off the intact pass at block band_start -- no extra forward needed.
    If FFN-off ~= frozen -> H2. If FFN-off > frozen -> attention in the band adds
    signal that the FFN normally destroys -> H1, stronger than the note claimed.
  * FIXED-AXIS readout: the note refit the axis on ablated states ("is anything
    still readable"). Here we ALSO fit the axis on intact train states and read
    the ablated test states with it ("is the original direction preserved").
  * Norm accounting: if ||f_l|| >> ||a_l|| in the band, 'attn-off does nothing'
    is expected regardless of mechanism, and stops being evidence.

Defaults follow the note: CounterFact 250 pairs, band 16-18, readout 18, fp32.
Conventions: 'block b' readout = hidden_states[b+1], as in ablation.py.
Reuses truth_probe.py (data, axis, CV) and anatomy.py (component hooks).
"""
import argparse
import torch
import truth_probe as T
import anatomy as A


# =====================================================================
#  shared helpers (pure; unit-testable without a model)
# =====================================================================
def test_indices(pidx, pair_list):
    """Flatten pairs -> (indices, labels) in true,false order."""
    I, Y = [], []
    for p in pair_list:
        it, iff = pidx[p]
        I += [it, iff]; Y += [1, 0]
    return I, torch.tensor(Y)


def heldout_auc_refit(H_states, pidx, folds, seed):
    """AUC with the axis REFIT on the same condition's train states (old protocol)."""
    aucs = []
    for tr, te in T.kfold_pairs(len(pidx), folds, seed):
        ax = T.fit_axis(H_states, [pidx[p] for p in tr])
        I, Y = test_indices(pidx, te)
        aucs.append(T.auc_score(T.project_fields(H_states[I], ax)["Re"], Y))
    return sum(aucs) / len(aucs)


def heldout_auc_fixed(H_axis_source, H_eval, pidx, folds, seed):
    """AUC with the axis fit on TRAIN states of H_axis_source (e.g. intact),
    evaluated on TEST states of H_eval (e.g. ablated). Same folds, no leakage."""
    aucs = []
    for tr, te in T.kfold_pairs(len(pidx), folds, seed):
        ax = T.fit_axis(H_axis_source, [pidx[p] for p in tr])
        I, Y = test_indices(pidx, te)
        aucs.append(T.auc_score(T.project_fields(H_eval[I], ax)["Re"], Y))
    return sum(aucs) / len(aucs)


def heldout_contrib_gap(H_axis_source, H_contrib, pidx, folds, seed):
    """Class gap of a CONTRIBUTION on the fixed truth axis, held out.
    Fit v1 on train states of H_axis_source; on test contributions compute
        gap  = mean(v1 . c | true) - mean(v1 . c | false)
        d'   = gap / pooled std   (effect size, scale-free)
    Returns (mean gap, mean d') across folds."""
    gaps, ds = [], []
    for tr, te in T.kfold_pairs(len(pidx), folds, seed):
        ax = T.fit_axis(H_axis_source, [pidx[p] for p in tr])
        I, Y = test_indices(pidx, te)
        proj = H_contrib[I].float() @ ax["v1"]
        pt, pf = proj[Y == 1], proj[Y == 0]
        gap = float(pt.mean() - pf.mean())
        pooled = float(torch.sqrt((pt.var() + pf.var()) / 2).clamp_min(1e-8))
        gaps.append(gap); ds.append(gap / pooled)
    return sum(gaps) / len(gaps), sum(ds) / len(ds)


def axis_full(H_states, pidx):
    """Descriptive axis fit on ALL pairs (used only for cosines/lens, labeled Obs.)."""
    return T.fit_axis(H_states, pidx)


# =====================================================================
#  forward passes with ablation (generalizes ablation.py: per-layer choice)
# =====================================================================
def collect_states(model, tok, items, dev, hs_indices, ablate_map=None, tag="run"):
    """One forward per sentence. ablate_map: {layer_idx: 'attn'|'ffn'|'both'} or None.
    Returns {hs_index: H [N, d]} for each requested hidden_states index."""
    layers = model.model.layers
    handles = []

    def attn_zero(_m, _i, out):
        if isinstance(out, tuple):
            return (torch.zeros_like(out[0]),) + tuple(out[1:])
        return torch.zeros_like(out)

    def ffn_zero(_m, _i, out):
        return torch.zeros_like(out)

    if ablate_map:
        for i, what in ablate_map.items():
            if what in ("attn", "both"):
                handles.append(layers[i].self_attn.register_forward_hook(attn_zero))
            if what in ("ffn", "both"):
                handles.append(layers[i].mlp.register_forward_hook(ffn_zero))

    out = {k: [] for k in hs_indices}
    try:
        for n, (_, txt) in enumerate(items):
            ids = tok(txt, return_tensors="pt").to(dev)
            with torch.no_grad():
                hs = model(**ids, output_hidden_states=True).hidden_states
            for k in hs_indices:
                out[k].append(hs[k][0, -1, :].detach().float().cpu())
            if (n + 1) % 20 == 0 or n + 1 == len(items):
                print(f"\r  [{tag:>12}] {n+1}/{len(items)}", end="", flush=True)
        print()
    finally:
        for h in handles:
            h.remove()
    return {k: torch.stack(v, 0) for k, v in out.items()}


def load_all(a):
    dev, _ = T.resolve_device_dtype("auto", "auto")
    dt = torch.float32
    pairs = list(T.DEFAULT_PAIRS) if a.dataset == "builtin" else \
        T.load_pairs(a.dataset, a.max_pairs, a.seed, a.rev_counterfact,
                     a.rev_truthfulqa, a.file_counterfact, a.file_truthfulqa)
    items, pidx = T.pairs_to_items(pairs)
    print(f"[data] {len(items)} sentences = {len(pidx)} pairs")
    print(f"[model] {a.model} on {dev} ({dt})")
    tok, model = T.load_model(a.model, dt, dev)
    return dev, tok, model, items, pidx


# =====================================================================
#  TASK contrib -- norms + class gap of the contributions on the fixed axis
# =====================================================================
def cmd_contrib(a):
    print("[task contrib] who writes how much, and WHAT, along the truth axis")
    print(f"[note] fixed axis = intact residual @block {a.axis_block} "
          f"(hidden_states[{a.axis_block+1}]), fit on train folds only")
    dev, tok, model, items, pidx = load_all(a)

    H_resid, H_attn, H_ffn = A.collect_components(model, tok, items, dev)
    rel = A.identity_check(H_resid, H_attn, H_ffn)
    print(f"[identity check] median {float(rel.median()):.2e}  (must be ~0)")
    if float(rel.median()) > 1e-3:
        print("  ABORT: decomposition invalid."); return

    Hax = H_resid[:, a.axis_block + 1, :]
    L = H_attn.shape[1]
    lo, hi = a.scan_start, min(a.scan_end, L - 1)

    print(f"\n{'layer':>5} | {'||attn||':>8} {'||ffn||':>8} | "
          f"{'gap_attn':>9} {'d\'_a':>6} | {'gap_ffn':>9} {'d\'_f':>6} | {'resid gap d\'':>12}")
    print("-" * 84)
    for Lb in range(lo, hi + 1):
        na = float(H_attn[:, Lb, :].norm(dim=-1).mean())
        nf = float(H_ffn[:, Lb, :].norm(dim=-1).mean())
        ga, da = heldout_contrib_gap(Hax, H_attn[:, Lb, :], pidx, a.folds, a.seed)
        gf, df = heldout_contrib_gap(Hax, H_ffn[:, Lb, :], pidx, a.folds, a.seed)
        _, dr = heldout_contrib_gap(Hax, H_resid[:, Lb + 1, :], pidx, a.folds, a.seed)
        band = "<- band" if a.band_start <= Lb <= a.band_end else ""
        print(f"{Lb:>5} | {na:>8.1f} {nf:>8.1f} | {ga:>+9.3f} {da:>+6.2f} | "
              f"{gf:>+9.3f} {df:>+6.2f} | {dr:>+12.2f} {band}")

    print("\n=== reading guide ===")
    print("  gap_x = mean(v1.x | true) - mean(v1.x | false), held out; d' = gap/pooled std.")
    print("  * gap_ffn NEGATIVE in the band  -> the FFN writes ANTI-truth along the axis")
    print("    (directional erosion, H1 strong).")
    print("  * gap_ffn ~0 with large ||ffn|| -> the FFN writes class-blind content; on a")
    print("    fixed axis that alone does NOT erode Re -- if ablation still helps, the")
    print("    erosion must come through refit/rotation or downstream interaction.")
    print("  * ||ffn|| >> ||attn|| in the band -> 'attn-off does nothing' was expected")
    print("    on norm grounds alone and is weak evidence by itself.")


# =====================================================================
#  TASK ablate -- refit vs fixed-axis vs frozen, one consistent run
# =====================================================================
def cmd_ablate(a):
    band = list(range(a.band_start, a.band_end + 1))
    ro = a.readout
    print("[task ablate] corrected ablation: refit vs FIXED axis vs FROZEN control")
    print(f"[note] band = layers {band}, readout = block {ro} (hidden_states[{ro+1}])")
    dev, tok, model, items, pidx = load_all(a)

    print("\n[three passes]")
    # intact: read both the band entrance (frozen equivalent) and the readout
    intact = collect_states(model, tok, items, dev,
                            hs_indices=[a.band_start, ro + 1], tag="intact")
    H_int_entry = intact[a.band_start]      # == frozen(both-off) state at readout
    H_int_ro = intact[ro + 1]
    ffnoff = collect_states(model, tok, items, dev, hs_indices=[ro + 1],
                            ablate_map={i: "ffn" for i in band}, tag="ffn-off")[ro + 1]
    attoff = collect_states(model, tok, items, dev, hs_indices=[ro + 1],
                            ablate_map={i: "attn" for i in band}, tag="attn-off")[ro + 1]

    f, s = a.folds, a.seed
    # 1) refit protocol (replicates the note)
    r_int = heldout_auc_refit(H_int_ro, pidx, f, s)
    r_frz = heldout_auc_refit(H_int_entry, pidx, f, s)
    r_ffn = heldout_auc_refit(ffnoff, pidx, f, s)
    r_att = heldout_auc_refit(attoff, pidx, f, s)
    # 2) fixed axis fit on intact READOUT states
    x_int = heldout_auc_fixed(H_int_ro, H_int_ro, pidx, f, s)   # == r_int by construction
    x_ffn = heldout_auc_fixed(H_int_ro, ffnoff, pidx, f, s)
    x_att = heldout_auc_fixed(H_int_ro, attoff, pidx, f, s)
    # 3) fixed axis fit on intact BAND-ENTRY states (the peak axis)
    p_ffn = heldout_auc_fixed(H_int_entry, ffnoff, pidx, f, s)
    p_att = heldout_auc_fixed(H_int_entry, attoff, pidx, f, s)
    p_int = heldout_auc_fixed(H_int_entry, H_int_ro, pidx, f, s)

    # descriptive cosines (full fit, Obs.)
    ax_entry = axis_full(H_int_entry, pidx)
    ax_ro = axis_full(H_int_ro, pidx)
    ax_ffn = axis_full(ffnoff, pidx)
    c_rot = abs(float(torch.dot(ax_entry["v1"], ax_ro["v1"])))
    c_abl = abs(float(torch.dot(ax_ro["v1"], ax_ffn["v1"])))
    c_ent = abs(float(torch.dot(ax_entry["v1"], ax_ffn["v1"])))

    print(f"\n=== held-out truth AUC @block {ro}, band {band} ===")
    print(f"{'condition':<26} {'refit':>7} {'fixed@readout':>14} {'fixed@entry':>12}")
    print(f"{'intact @entry (=FROZEN)':<26} {r_frz:>7.3f} {'-':>14} {'-':>12}")
    print(f"{'intact @readout':<26} {r_int:>7.3f} {x_int:>14.3f} {p_int:>12.3f}")
    print(f"{'FFN-off in band':<26} {r_ffn:>7.3f} {x_ffn:>14.3f} {p_ffn:>12.3f}")
    print(f"{'attn-off in band':<26} {r_att:>7.3f} {x_att:>14.3f} {p_att:>12.3f}")
    print(f"\n[axis geometry, descriptive] |cos(v1_entry, v1_readout)| = {c_rot:.3f}   "
          f"|cos(v1_readout, v1_ffnoff)| = {c_abl:.3f}   |cos(v1_entry, v1_ffnoff)| = {c_ent:.3f}")

    print("\n=== verdict logic ===")
    d = r_ffn - r_frz
    if abs(d) <= 0.02:
        print(f"  FFN-off ({r_ffn:.3f}) ~= FROZEN ({r_frz:.3f}) [diff {d:+.3f}]:")
        print("  -> H2. Zeroing the FFN is equivalent to freezing the stream at the band")
        print("     entrance. The 'erosion' is just 'the FFN is the only writer after the")
        print("     peak'; nothing specifically anti-truth about what it writes.")
    elif d > 0.02:
        print(f"  FFN-off ({r_ffn:.3f}) BEATS FROZEN ({r_frz:.3f}) [diff {d:+.3f}]:")
        print("  -> H1+. With the FFN silenced, attention in the band ADDS signal beyond")
        print("     the peak; normally the FFN destroys it. Active, directional erosion.")
    else:
        print(f"  FFN-off ({r_ffn:.3f}) BELOW FROZEN ({r_frz:.3f}) [diff {d:+.3f}]:")
        print("  -> partial erosion also from attention/interaction: silencing the FFN")
        print("     does not even recover the band-entry signal.")
    print(f"  fixed-axis check: intact axis reads FFN-off states at {x_ffn:.3f} "
          f"(refit {r_ffn:.3f});")
    print("  small difference -> the DIRECTION is preserved, erosion acts along the axis;")
    print("  large difference -> ablation rotates the readable direction.")


# =====================================================================
#  TASK sweep -- which layer's FFN erodes?
# =====================================================================
def cmd_sweep(a):
    band = list(range(a.band_start, a.band_end + 1))
    ro = a.readout
    print("[task sweep] FFN-off one layer at a time (+ cumulative), same readout")
    print(f"[note] band = {band}, readout = block {ro}")
    dev, tok, model, items, pidx = load_all(a)
    f, s = a.folds, a.seed

    print("\n[passes]")
    H_int = collect_states(model, tok, items, dev, [ro + 1], None, "intact")[ro + 1]
    base = heldout_auc_refit(H_int, pidx, f, s)
    print(f"\n  intact @block {ro}: {base:.3f}\n")
    print(f"{'FFN zeroed at':<18} {'refit AUC':>9} {'vs intact':>10}")
    print("-" * 42)
    results = []
    for Lb in band:
        H = collect_states(model, tok, items, dev, [ro + 1],
                           {Lb: "ffn"}, f"ffn-off {Lb}")[ro + 1]
        auc = heldout_auc_refit(H, pidx, f, s)
        results.append((f"{{{Lb}}}", auc))
        print(f"{'{'+str(Lb)+'}':<18} {auc:>9.3f} {auc-base:>+10.3f}")
    for j in range(2, len(band) + 1):
        sub = band[:j]
        H = collect_states(model, tok, items, dev, [ro + 1],
                           {i: "ffn" for i in sub}, f"ffn-off {sub}")[ro + 1]
        auc = heldout_auc_refit(H, pidx, f, s)
        results.append((str(set(sub)), auc))
        print(f"{str(set(sub)):<18} {auc:>9.3f} {auc-base:>+10.3f}")

    best = max(results, key=lambda r: r[1])
    print(f"\n  largest recovery: FFN-off at {best[0]} -> {best[1]:.3f} ({best[1]-base:+.3f})")
    print("  concentrated in one layer -> a specific FFN is the eroder (mechanistically")
    print("  attackable: lens it, then neuron-level later). Spread across the band ->")
    print("  distributed drift toward next-token content.")


# =====================================================================
#  TASK lens -- qualitative: what tokens do the band FFNs promote?
# =====================================================================
def cmd_lens(a):
    band = list(range(a.band_start, a.band_end + 1))
    print("[task lens] logit-lens of the FFN contributions in the band (QUALITATIVE, Obs.)")
    print("[caveat] applying final-norm + unembedding to a mid-stack CONTRIBUTION is the")
    print("         standard logit-lens heuristic, not an exact readout. Directional hints only.")
    dev, tok, model, items, pidx = load_all(a)

    H_resid, H_attn, H_ffn = A.collect_components(model, tok, items, dev)
    y = torch.tensor([l for l, _ in items])
    Hax = H_resid[:, a.axis_block + 1, :]
    ax = axis_full(Hax, pidx)   # descriptive axis for projections

    W = model.get_output_embeddings().weight.detach()      # [V, d]
    fnorm = model.model.norm

    def top_tokens(vec, k):
        with torch.no_grad():
            v = fnorm(vec.to(dev).to(next(model.parameters()).dtype))
            logits = (W.to(v.dtype) @ v).float().cpu()
        vals, ids = logits.topk(k)
        toks = [tok.decode([int(i)]).replace("\n", "\\n") for i in ids]
        return list(zip(toks, [float(x) for x in vals]))

    for Lb in band:
        Fb = H_ffn[:, Lb, :]
        mt = Fb[y == 1].mean(0); mf = Fb[y == 0].mean(0)
        pt = float(mt @ ax["v1"]); pf = float(mf @ ax["v1"])
        print(f"\n--- FFN layer {Lb} ---")
        print(f"  mean contribution . v1 :  true {pt:+.3f}   false {pf:+.3f}   "
              f"(class gap {pt-pf:+.3f})")
        print(f"  top tokens (mean over TRUE):  "
              + ", ".join(f"{t!r}" for t, _ in top_tokens(mt, a.topk)))
        print(f"  top tokens (mean over FALSE): "
              + ", ".join(f"{t!r}" for t, _ in top_tokens(mf, a.topk)))
        diff = mt - mf
        print(f"  top tokens of (TRUE - FALSE): "
              + ", ".join(f"{t!r}" for t, _ in top_tokens(diff, a.topk)))
        print(f"  top tokens of (FALSE - TRUE): "
              + ", ".join(f"{t!r}" for t, _ in top_tokens(-diff, a.topk)))
    print("\n  reading: format/punctuation/template tokens dominating both classes ->")
    print("  supports 'the FFN shifts the stream toward next-token/format content'.")
    print("  Semantically negating tokens in FALSE-TRUE -> content-level interference.")


# =====================================================================
#  CLI
# =====================================================================
def add_common(p):
    p.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
    p.add_argument("--dataset", default="counterfact",
                   choices=["builtin", "counterfact", "truthfulqa", "mix"])
    p.add_argument("--max-pairs", type=int, default=250)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--band-start", type=int, default=16)
    p.add_argument("--band-end", type=int, default=18)
    p.add_argument("--rev-counterfact", default="c945b082ca08d0a8f3ba227fb78404a09614c36e")
    p.add_argument("--rev-truthfulqa", default="741b8276f2d1982aa3d5b832d3ee81ed3b896490")
    p.add_argument("--file-counterfact", default=None)
    p.add_argument("--file-truthfulqa", default=None)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="task", required=True)

    pc = sub.add_parser("contrib", help="norms + class gap of attn/ffn contributions on fixed axis")
    add_common(pc)
    pc.add_argument("--axis-block", type=int, default=15, help="block whose intact output defines v1")
    pc.add_argument("--scan-start", type=int, default=10)
    pc.add_argument("--scan-end", type=int, default=22)
    pc.set_defaults(func=cmd_contrib)

    pa = sub.add_parser("ablate", help="refit vs fixed-axis vs frozen control")
    add_common(pa)
    pa.add_argument("--readout", type=int, default=18)
    pa.set_defaults(func=cmd_ablate)

    ps = sub.add_parser("sweep", help="FFN-off one layer at a time + cumulative")
    add_common(ps)
    ps.add_argument("--readout", type=int, default=18)
    ps.set_defaults(func=cmd_sweep)

    pl = sub.add_parser("lens", help="qualitative logit-lens of the band FFN contributions")
    add_common(pl)
    pl.add_argument("--axis-block", type=int, default=15)
    pl.add_argument("--topk", type=int, default=10)
    pl.set_defaults(func=cmd_lens)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
