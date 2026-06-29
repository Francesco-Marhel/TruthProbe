# -*- coding: utf-8 -*-
"""
ablation.py  --  Causal test: is a component NECESSARY for the truth signal,
or just where the signal is most readable?

The consolidation showed attention dominates the truth axis at the middle/peak
layers and the FFN at the late layers (a stable 'passa-parola'). But dominance
in READABILITY is correlational: the signal could be produced elsewhere and
merely read there. Ablation tests CAUSE. We zero a component's contribution
(hook returns zeros -> residual + 0 = residual, the component adds nothing),
let the effect propagate downstream, and measure how much the truth signal at a
readout layer COLLAPSES versus the intact model.

We ablate the STRONG component (the one that dominates), not the weak one:
the question is whether the dominant component is causally required. Two
conditions, directly comparable: ATTN-off and FFN-off over the same band, same
readout. A larger collapse = that component causally builds the signal there.

The 'passa-parola' becomes testable: if attention builds the signal in the
middle band, ablating attention in the middle band should collapse the peak
signal more than ablating the FFN there.

Reuses truth_probe.py. fp32. Observational metric = held-out truth AUC, refit on
the ABLATED states (does truth remain readable at all from what survives?).
"""
import argparse
import torch
import truth_probe as T


def collect_ablated(model, tok, items, dev, readout, ablate, band):
    """Run forward with `ablate` ('attn'|'ffn'|None) zeroed on layers in `band`.
    Return last-token residual state at block `readout` -> H [N, d]."""
    layers = model.model.layers
    handles = []

    def mk_attn_zero():
        def hook(_m, _i, out):
            if isinstance(out, tuple):
                return (torch.zeros_like(out[0]),) + tuple(out[1:])
            return torch.zeros_like(out)
        return hook

    def mk_ffn_zero():
        def hook(_m, _i, out):
            return torch.zeros_like(out)
        return hook

    if ablate is not None:
        for i in band:
            mod = layers[i].self_attn if ablate == "attn" else layers[i].mlp
            handles.append(mod.register_forward_hook(
                mk_attn_zero() if ablate == "attn" else mk_ffn_zero()))

    H = []
    try:
        for n, (_, txt) in enumerate(items):
            ids = tok(txt, return_tensors="pt").to(dev)
            with torch.no_grad():
                out = model(**ids, output_hidden_states=True)
            H.append(out.hidden_states[readout + 1][0, -1, :].detach().float().cpu())
            if (n + 1) % 20 == 0 or n + 1 == len(items):
                print(f"\r  [{ablate or 'intact':>6}] {n+1}/{len(items)}", end="", flush=True)
        print()
    finally:
        for h in handles:
            h.remove()
    return torch.stack(H, 0)


def auc(Hc, pidx, folds, seed):
    aucs = []
    for tr, te in T.kfold_pairs(len(pidx), folds, seed):
        ax = T.fit_axis(Hc, [pidx[p] for p in tr])
        I, Y = [], []
        for p in te:
            it, iff = pidx[p]; I += [it, iff]; Y += [1, 0]
        Re = T.project_fields(Hc[I], ax)["Re"]
        aucs.append(T.auc_score(Re, torch.tensor(Y)))
    return sum(aucs) / len(aucs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
    ap.add_argument("--dataset", default="builtin",
                    choices=["builtin", "counterfact", "truthfulqa", "mix"])
    ap.add_argument("--max-pairs", type=int, default=250)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--readout", type=int, default=15, help="block whose output we read truth from")
    ap.add_argument("--band-start", type=int, default=11, help="first layer to ablate")
    ap.add_argument("--band-end", type=int, default=15, help="last layer to ablate (inclusive)")
    ap.add_argument("--rev-counterfact", default=None)
    ap.add_argument("--rev-truthfulqa", default=None)
    ap.add_argument("--file-counterfact", default=None)
    ap.add_argument("--file-truthfulqa", default=None)
    a = ap.parse_args()

    dev, _ = T.resolve_device_dtype("auto", "auto")
    dt = torch.float32
    band = list(range(a.band_start, a.band_end + 1))
    print("[task ablation] is the dominant component CAUSALLY necessary for truth?")
    print(f"[note] fp32; ablate band = layers {band}; readout = block {a.readout}")
    pairs = list(T.DEFAULT_PAIRS) if a.dataset == "builtin" else \
        T.load_pairs(a.dataset, a.max_pairs, a.seed, a.rev_counterfact,
                     a.rev_truthfulqa, a.file_counterfact, a.file_truthfulqa)
    items, pidx = T.pairs_to_items(pairs)
    print(f"[data] {len(items)} sentences = {len(pidx)} pairs")
    print(f"[model] {a.model} on {dev} ({dt})")
    tok, model = T.load_model(a.model, dt, dev)

    print("\n[running three forward passes: intact, attention-off, FFN-off]")
    H_intact = collect_ablated(model, tok, items, dev, a.readout, None, band)
    H_no_attn = collect_ablated(model, tok, items, dev, a.readout, "attn", band)
    H_no_ffn = collect_ablated(model, tok, items, dev, a.readout, "ffn", band)

    auc_intact = auc(H_intact, pidx, a.folds, a.seed)
    auc_no_attn = auc(H_no_attn, pidx, a.folds, a.seed)
    auc_no_ffn = auc(H_no_ffn, pidx, a.folds, a.seed)
    drop_attn = auc_intact - auc_no_attn
    drop_ffn = auc_intact - auc_no_ffn

    print(f"\n=== truth AUC at block {a.readout} (ablating layers {band}) ===")
    print(f"  intact                 : {auc_intact:.3f}")
    print(f"  attention OFF in band  : {auc_no_attn:.3f}   (drop {drop_attn:+.3f})")
    print(f"  FFN       OFF in band  : {auc_no_ffn:.3f}   (drop {drop_ffn:+.3f})")
    print("\n=== causal reading ===")
    print("  A larger drop = that component is more causally necessary for the truth")
    print("  signal at the readout. (Readability != necessity; this measures necessity.)")
    if abs(drop_attn - drop_ffn) < 0.03:
        print("  -> attention and FFN are about equally necessary here: no causal asymmetry,")
        print("     despite the readability difference. The 'who carries it' question is")
        print("     not resolved by removal -- both contribute to the surviving signal.")
    elif drop_attn > drop_ffn:
        print("  -> removing ATTENTION hurts more: attention is causally building the signal")
        print("     in this band, not merely the place it is most readable.")
    else:
        print("  -> removing the FFN hurts more: the FFN is causally building the signal here,")
        print("     even if attention read higher. Readability misled; necessity is the FFN.")


if __name__ == "__main__":
    main()
