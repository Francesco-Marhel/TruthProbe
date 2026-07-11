"""
axis_norm_check.py -- is the truth axis an artifact of norm weighting?

fit_axis applies SVD to raw difference rows, so v1 weights pair i by
||d_i||^2. This tool refits on row-normalized differences and reports:
  (1) cos(v1_raw, v1_unit)      -- do the two axes agree?
  (2) held-out AUC of both      -- same k-fold protocol as truth_probe
  (3) corr(||d_i||, token length of the true sentence)

Run from src/:  python axis_norm_check.py --model Qwen/Qwen2.5-3B --layer 17
(--layer is the hidden level = peak block + 1)
"""
import argparse, torch
from truth_probe import (load_model, load_pairs, pairs_to_items, collect,
                         resolve_device_dtype, auc_score, kfold_pairs)

REV = "c945b082ca08d0a8f3ba227fb78404a09614c36e"

def fit_v1(D):
    _, _, Vh = torch.linalg.svd(D, full_matrices=False)
    return Vh[0].clone()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--layer", type=int, required=True,
                    help="hidden level (= peak block + 1)")
    ap.add_argument("--max-pairs", type=int, default=250)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--pool", default="last")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--rev-counterfact", default=REV)
    a = ap.parse_args()

    dev, dt = resolve_device_dtype(a.device, a.dtype)
    pairs = load_pairs("counterfact", a.max_pairs, a.seed,
                       a.rev_counterfact, None, None, None)
    items, pidx = pairs_to_items(pairs)
    print(f"[data] {len(items)} sentences = {len(pidx)} pairs")
    tok, model = load_model(a.model, dt, dev)
    H = collect(model, tok, items, dev, a.pool)
    Hl = H[:, a.layer, :].float()

    t_idx = [t for t, f in pidx]; f_idx = [f for t, f in pidx]
    D = Hl[t_idx] - Hl[f_idx]
    norms = D.norm(dim=1)
    lens = torch.tensor([len(tok(items[t][1]).input_ids) for t, f in pidx],
                        dtype=torch.float32)
    r = torch.corrcoef(torch.stack([norms, lens]))[0, 1].item()
    print(f"\n||d_i||: mean {norms.mean():.2f}  min {norms.min():.2f}  "
          f"max {norms.max():.2f}   corr(||d||, token length) = {r:+.3f}")

    v_raw, v_unit = fit_v1(D), fit_v1(D / norms.unsqueeze(1))
    cos = torch.dot(v_raw / v_raw.norm(), v_unit / v_unit.norm()).abs().item()
    print(f"|cos(v1_raw, v1_unit)| = {cos:.4f}")

    y = torch.tensor([1.0] * len(t_idx) + [0.0] * len(f_idx))
    states = torch.cat([Hl[t_idx], Hl[f_idx]], 0)
    for name, normalize in (("raw SVD", False), ("row-normalized SVD", True)):
        oof = torch.full((len(y),), float("nan"))
        for tr, te in kfold_pairs(len(pidx), a.folds, a.seed):
            Dtr = D[tr]
            v = fit_v1(Dtr / Dtr.norm(dim=1, keepdim=True) if normalize else Dtr)
            sc = states @ v
            if sc[: len(t_idx)][tr].mean() < sc[len(t_idx):][tr].mean():
                sc = -sc
            for i in te:
                oof[i] = sc[i]; oof[len(t_idx) + i] = sc[len(t_idx) + i]
        m = ~torch.isnan(oof)
        print(f"held-out AUC ({name}): {auc_score(oof[m], y[m]):.3f}")
    print("\nreading: |cos| near 1 and matching AUCs = the axis is not a "
          "norm-weighting artifact; a gap is itself a finding.")

if __name__ == "__main__":
    main()
