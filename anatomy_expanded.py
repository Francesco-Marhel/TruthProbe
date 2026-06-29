# -*- coding: utf-8 -*-
"""
anatomy_expanded.py  --  Is the truth axis more readable INSIDE the FFN's
expanded activation (post-SwiGLU, pre-down-projection) than in the compressed
residual stream?

In Qwen2 the MLP computes:
    down_proj( act_fn(gate_proj(x)) * up_proj(x) )
The term inside down_proj -- act_fn(gate_proj(x)) * up_proj(x) -- is the EXPANDED
activation, of size intermediate_size (8960 for Qwen2.5-1.5B). It is the sparse,
gonfiato space where SwiGLU works, BEFORE the down-projection re-compresses it
back into the 1536-dim residual stream. We capture it with a forward_pre_hook on
down_proj (whose input IS that expanded activation) and fit the truth axis there.

*** CRITICAL CAVEAT, built into the verdict ***
The expanded space has 8960 dimensions but we have only tens-to-hundreds of pairs.
In this p >> n regime the SVD can separate ANY labels in-sample by chance -- so a
high held-out AUC is NOT automatically meaningful, and the permutation null will
sit WELL above 0.5. Only the MARGIN of the real AUC over the (elevated) null is
real signal. The script reports the null prominently; do not read the raw AUC.

Reuses truth_probe.py for data + geometry. fp32 (precision/consistency).
"""
import argparse
import random
import torch
import truth_probe as T


def collect_expanded(model, tok, items, dev):
    """Returns last-token states:
       H_resid [N, L+1, d_model]   the residual stream (reference)
       H_exp   [N, L,   d_inter]   the FFN expanded activation (input to down_proj)"""
    layers = model.model.layers
    L = len(layers)
    buf = {}

    def mk_pre_hook(name):
        # forward_pre_hook on down_proj: args[0] is the expanded activation feeding it
        def hook(_module, args):
            x = args[0]
            buf[name] = x[0, -1, :].detach().float().cpu()
        return hook

    handles = []
    for i, layer in enumerate(layers):
        handles.append(layer.mlp.down_proj.register_forward_pre_hook(mk_pre_hook(f"exp{i}")))

    H_resid, H_exp = [], []
    try:
        for n, (_, txt) in enumerate(items):
            buf.clear()
            ids = tok(txt, return_tensors="pt").to(dev)
            with torch.no_grad():
                out = model(**ids, output_hidden_states=True)
            hs = out.hidden_states
            resid = torch.stack([h[0, -1, :].detach().float().cpu() for h in hs], 0)
            exp = torch.stack([buf[f"exp{i}"] for i in range(L)], 0)
            H_resid.append(resid); H_exp.append(exp)
            if (n + 1) % 10 == 0 or n + 1 == len(items):
                print(f"\r[extract] {n+1}/{len(items)} sentences", end="", flush=True)
        print()
    finally:
        for h in handles:
            h.remove()
    return torch.stack(H_resid, 0), torch.stack(H_exp, 0)


def auc_component(Hc, pidx, folds, seed):
    aucs = []
    for tr, te in T.kfold_pairs(len(pidx), folds, seed):
        ax = T.fit_axis(Hc, [pidx[p] for p in tr])
        I, Y = [], []
        for p in te:
            it, iff = pidx[p]; I += [it, iff]; Y += [1, 0]
        Re = T.project_fields(Hc[I], ax)["Re"]
        aucs.append(T.auc_score(Re, torch.tensor(Y)))
    return sum(aucs) / len(aucs)


def perm_null(Hc, pidx, folds, seed, nperm):
    rng = random.Random(seed); null = []
    for b in range(nperm):
        o = [1 if rng.random() < 0.5 else -1 for _ in pidx]
        aucs = []
        for tr, te in T.kfold_pairs(len(pidx), folds, seed):
            ax = T.fit_axis(Hc, [pidx[p] for p in tr], [o[p] for p in tr])
            I, Y = [], []
            for p in te:
                it, iff = pidx[p]
                if o[p] >= 0: I += [it, iff]; Y += [1, 0]
                else:         I += [iff, it]; Y += [1, 0]
            Re = T.project_fields(Hc[I], ax)["Re"]
            aucs.append(T.auc_score(Re, torch.tensor(Y)))
        null.append(sum(aucs) / len(aucs))
        print(f"\r[permutation] {b+1}/{nperm}", end="", flush=True)
    print()
    return torch.tensor(null)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
    ap.add_argument("--dataset", default="builtin",
                    choices=["builtin", "counterfact", "truthfulqa", "mix"])
    ap.add_argument("--max-pairs", type=int, default=250)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--perm", type=int, default=100,
                    help="permutation reps at the peak (REQUIRED for honesty in high-dim)")
    ap.add_argument("--rev-counterfact", default=None)
    ap.add_argument("--rev-truthfulqa", default=None)
    ap.add_argument("--file-counterfact", default=None)
    ap.add_argument("--file-truthfulqa", default=None)
    a = ap.parse_args()

    dev, _ = T.resolve_device_dtype("auto", "auto")
    dt = torch.float32
    print("[task anatomy_expanded] is truth more readable in the FFN's expanded space?")
    print("[note] fp32; expanded dim is large (p>>n) -- trust the MARGIN over the null, not the raw AUC")
    pairs = list(T.DEFAULT_PAIRS) if a.dataset == "builtin" else \
        T.load_pairs(a.dataset, a.max_pairs, a.seed, a.rev_counterfact,
                     a.rev_truthfulqa, a.file_counterfact, a.file_truthfulqa)
    items, pidx = T.pairs_to_items(pairs)
    print(f"[data] {len(items)} sentences = {len(pidx)} pairs")
    print(f"[model] {a.model} on {dev} ({dt})")
    tok, model = T.load_model(a.model, dt, dev)

    H_resid, H_exp = collect_expanded(model, tok, items, dev)
    L = H_exp.shape[1]
    d_inter = H_exp.shape[2]
    print(f"[dims] residual {H_resid.shape[2]}  |  FFN expanded {d_inter}  "
          f"(pairs={len(pidx)}  ->  p>>n: {d_inter} dims vs {len(pidx)} pairs)\n")

    print(f"{'layer':>5} | {'resid':>7} | {'FFN-expanded':>12}")
    print("-" * 32)
    rows = []
    for Lb in range(L):
        auc_r = auc_component(H_resid[:, Lb + 1, :], pidx, a.folds, a.seed)
        auc_e = auc_component(H_exp[:, Lb, :],       pidx, a.folds, a.seed)
        rows.append((Lb, auc_r, auc_e))
        print(f"{Lb:>5} | {auc_r:>7.3f} | {auc_e:>12.3f}")

    peak = max(rows, key=lambda r: r[1])
    Lb, auc_r, auc_e = peak
    print(f"\n=== peak residual layer = block {Lb} (residual AUC {auc_r:.3f}) ===")
    print(f"  FFN-expanded axis AUC : {auc_e:.3f}")

    if a.perm > 0:
        print("\n[peak permutation null on the EXPANDED axis] -- this is the real test")
        nt = perm_null(H_exp[:, Lb, :], pidx, a.folds, a.seed, a.perm)
        p = (1 + int((nt >= auc_e).sum())) / (len(nt) + 1)
        margin = auc_e - float(nt.quantile(0.95))
        print(f"  observed {auc_e:.3f}  null mean {nt.mean():.3f}  95pct {nt.quantile(0.95):.3f}  p={p:.4f}")
        print(f"  margin over null 95pct: {margin:+.3f}")
        if float(nt.mean()) > 0.6:
            print("  NOTE: the null itself is high -- confirms p>>n overfitting. The raw AUC is")
            print("  inflated by dimensionality; only the margin below counts.")
        if p < 0.05 and margin > 0.05:
            print("  -> real signal in the expanded space ABOVE the overfitting floor.")
            print("     Worth comparing to the residual/compressed axis: does expansion help?")
        elif p < 0.05:
            print("  -> barely above the inflated null: real but marginal. Treat with caution.")
        else:
            print("  -> NOT above the overfitting floor: the high AUC is dimensionality, not truth.")
            print("     The expanded space does NOT read truth better. Clean negative.")


if __name__ == "__main__":
    main()
