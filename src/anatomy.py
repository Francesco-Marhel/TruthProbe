# -*- coding: utf-8 -*-
"""
anatomy.py  --  Where is the truth axis BORN: attention or FFN?

This does NOT chase a better separation. It is an anatomy of the 1D truth axis
we already found. The residual stream we always studied is a SUM:
    h_after_block = h_before_block + attn_contribution + ffn_contribution
We read only the sum. Here we split it: hooks capture, per layer, the vector the
attention adds and the vector the FFN adds, separately. Then we run the SAME
unsupervised true/false SVD axis on each component and measure, per layer, how
much of the truth signal lives in the attention's contribution vs the FFN's.

Purpose: understand HOW the axis forms (mechanism), not WHERE it peaks (known).
Observational. Shield: held-out CV over pairs; optional permutation at the peak.

Correctness gate (printed): attn + ffn must equal the residual-stream delta
(h[L+1] - h[L]). If that identity does not hold (~0 error), the decomposition is
wrong and the numbers are meaningless. The script verifies it on real data first.

Reuses truth_probe.py for data + geometry, so the axis is identical to the paper.
"""
import argparse
import random
import torch
import truth_probe as T


def collect_components(model, tok, items, dev):
    """One forward pass per sentence. Returns last-token states:
       H_resid [N, L+1, d]  (the residual stream, embedding + each block output)
       H_attn  [N, L,   d]  (per block: the vector attention adds to the residual)
       H_ffn   [N, L,   d]  (per block: the vector the FFN adds to the residual)"""
    layers = model.model.layers
    L = len(layers)
    buf = {}

    def mk_hook(name):
        def hook(_module, _inp, out):
            o = out[0] if isinstance(out, tuple) else out   # self_attn returns a tuple
            buf[name] = o[0, -1, :].detach().float().cpu()
        return hook

    handles = []
    for i, layer in enumerate(layers):
        handles.append(layer.self_attn.register_forward_hook(mk_hook(f"attn{i}")))
        handles.append(layer.mlp.register_forward_hook(mk_hook(f"ffn{i}")))

    H_resid, H_attn, H_ffn = [], [], []
    try:
        for n, (_, txt) in enumerate(items):
            buf.clear()
            ids = tok(txt, return_tensors="pt").to(dev)
            with torch.no_grad():
                out = model(**ids, output_hidden_states=True)
            hs = out.hidden_states                          # tuple length L+1, each [1, seq, d]
            resid = torch.stack([h[0, -1, :].detach().float().cpu() for h in hs], 0)
            attn = torch.stack([buf[f"attn{i}"] for i in range(L)], 0)
            ffn = torch.stack([buf[f"ffn{i}"] for i in range(L)], 0)
            H_resid.append(resid); H_attn.append(attn); H_ffn.append(ffn)
            if (n + 1) % 10 == 0 or n + 1 == len(items):
                print(f"\r[extract] {n+1}/{len(items)} sentences", end="", flush=True)
        print()
    finally:
        for h in handles:
            h.remove()
    return torch.stack(H_resid, 0), torch.stack(H_attn, 0), torch.stack(H_ffn, 0)


def identity_check(H_resid, H_attn, H_ffn):
    """attn[i] + ffn[i] must equal resid[i+1] - resid[i].
    Returns per-(sentence,layer) relative error, robustly summarized. The MAX is
    fragile: where the true delta is near zero (early layers add almost nothing),
    any tiny absolute error blows up the ratio. The MEDIAN tells whether the
    decomposition is exact for the typical case."""
    L = H_attn.shape[1]
    delta = H_resid[:, 1:L + 1, :] - H_resid[:, 0:L, :]      # [N, L, d]
    recon = H_attn + H_ffn
    abs_err = (recon - delta).norm(dim=-1)                   # [N, L]
    scale = delta.norm(dim=-1).clamp_min(1e-6)
    rel = abs_err / scale                                    # [N, L]
    return rel


def auc_component(Hc, pidx, folds, seed):
    """Held-out AUC of the unsupervised truth axis on one component (per layer slice)."""
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
    """Permutation null for one component: swap true/false within pairs, refit."""
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
    ap.add_argument("--perm", type=int, default=0, help="permutation reps at the peak layer (0=skip)")
    ap.add_argument("--rev-counterfact", default=None)
    ap.add_argument("--rev-truthfulqa", default=None)
    ap.add_argument("--file-counterfact", default=None)
    ap.add_argument("--file-truthfulqa", default=None)
    a = ap.parse_args()

    dev, _ = T.resolve_device_dtype("auto", "auto")
    dt = torch.float32                      # REQUIRED: the attn+ffn==delta identity needs fp32.
                                            # In bf16 the residual-delta suffers catastrophic
                                            # cancellation (huge residual norms minus tiny deltas)
                                            # and the identity check fails spuriously. ~6GB in fp32.
    print("[task anatomy] where is the truth axis born: attention or FFN?")
    print("[note] loading in float32 (the decomposition identity needs the precision)")
    pairs = list(T.DEFAULT_PAIRS) if a.dataset == "builtin" else \
        T.load_pairs(a.dataset, a.max_pairs, a.seed, a.rev_counterfact,
                     a.rev_truthfulqa, a.file_counterfact, a.file_truthfulqa)
    items, pidx = T.pairs_to_items(pairs)
    print(f"[data] {len(items)} sentences = {len(pidx)} pairs")
    print(f"[model] {a.model} on {dev} ({dt})")
    tok, model = T.load_model(a.model, dt, dev)

    H_resid, H_attn, H_ffn = collect_components(model, tok, items, dev)
    L = H_attn.shape[1]

    rel = identity_check(H_resid, H_attn, H_ffn)             # [N, L]
    med = float(rel.median()); mx = float(rel.max())
    print(f"\n[identity check] (attn+ffn == residual delta), relative error")
    print(f"  median: {med:.2e}   max: {mx:.2e}")
    print("  per-layer median error (high only where the residual delta is tiny):")
    perlayer = rel.median(dim=0).values                      # [L]
    for Lb in range(0, len(perlayer), 7):
        chunk = perlayer[Lb:Lb + 7]
        print("   L%-2d: " % Lb + "  ".join(f"{float(v):.1e}" for v in chunk))
    if med < 1e-3:
        print("  OK: decomposition is EXACT for the typical layer. Numbers are trustworthy.")
        print("  (A large max, if any, comes from early layers whose delta is ~0 -> ratio blows up.)")
    else:
        print("  WARNING: the decomposition is wrong even for the typical layer.")
        print("  Numbers below are unreliable -- do NOT interpret them.")

    # per-layer held-out AUC: residual (after block), attention contribution, FFN contribution
    print(f"\n{'layer':>5} | {'resid':>7} | {'attn':>7} | {'ffn':>7}")
    print("-" * 36)
    rows = []
    for Lb in range(L):
        auc_r = auc_component(H_resid[:, Lb + 1, :], pidx, a.folds, a.seed)
        auc_a = auc_component(H_attn[:, Lb, :],      pidx, a.folds, a.seed)
        auc_f = auc_component(H_ffn[:, Lb, :],       pidx, a.folds, a.seed)
        rows.append((Lb, auc_r, auc_a, auc_f))
        print(f"{Lb:>5} | {auc_r:>7.3f} | {auc_a:>7.3f} | {auc_f:>7.3f}")

    peak = max(rows, key=lambda r: r[1])           # peak by residual AUC
    Lb, auc_r, auc_a, auc_f = peak
    print(f"\n=== peak residual layer = block {Lb} (residual AUC {auc_r:.3f}) ===")
    print(f"  attention contribution AUC : {auc_a:.3f}")
    print(f"  FFN contribution AUC       : {auc_f:.3f}")
    if auc_a > auc_f + 0.03:
        print("  -> the truth axis lives more in the ATTENTION's contribution here.")
    elif auc_f > auc_a + 0.03:
        print("  -> the truth axis lives more in the FFN's contribution here.")
    else:
        print("  -> attention and FFN carry the truth signal about equally here.")

    if a.perm > 0:
        print("\n[peak permutation null]")
        for name, Hc in [("attn", H_attn[:, Lb, :]), ("ffn", H_ffn[:, Lb, :])]:
            nt = perm_null(Hc, pidx, a.folds, a.seed, a.perm)
            obs = auc_a if name == "attn" else auc_f
            p = (1 + int((nt >= obs).sum())) / (len(nt) + 1)
            print(f"  {name}: observed {obs:.3f}  null mean {nt.mean():.3f}  "
                  f"95pct {nt.quantile(0.95):.3f}  p={p:.4f}")


if __name__ == "__main__":
    main()
