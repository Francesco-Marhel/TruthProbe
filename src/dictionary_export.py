# -*- coding: utf-8 -*-
"""
dictionary_export.py  --  The theorist's handoff: extract and SAVE the measured
truth dictionary, so the engineering step starts from a file, not a description.

Contents of the exported bundle (torch.save dict, plus a .json human summary):
  cats            : list of relation ids (K)
  templates       : human-readable relation templates
  axes            : [K, d]  per-category truth axes at the peak, unit norm,
                    oriented true = positive within their own category
  t_global        : [d]     global truth axis (all categories pooled), oriented
  write_centroids : [K, d]  unit centroids of the FFN's class-signed write
                    directions (Delta f at the write layer)
  cos_peak, cos_early : [K, K] SIGNED cosine matrices
  transfer        : [K, K]  held-out AUC transfer matrix at the peak
  meta            : model, peak block, write layer, dataset revision, seed,
                    pairs per category, decoding stats (Delta f vs lexical)

A K-direction readout built from `axes` (project the peak-block state on each
axis) is a truth/category monitor of K x d parameters -- the 'tiny module'
this bundle exists to enable. The registered engineering criterion: any
trained sparse dictionary must BEAT this measured one (class-signal
reconstruction and category decoding) to justify its parameters.

    python dictionary_export.py                          # Qwen2.5-1.5B
    python dictionary_export.py --model Qwen/Qwen2.5-3B --peak 16 --write-layer 17
"""
import argparse
import json
import torch
import truth_probe as T
import anatomy as A
from categories import (load_category_pairs, unit, axis_cosine_matrix,
                        transfer_matrix, decoding_with_null)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
    ap.add_argument("--k-relations", type=int, default=5)
    ap.add_argument("--pairs-per-relation", type=int, default=60)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--perm", type=int, default=100)
    ap.add_argument("--peak", type=int, default=15)
    ap.add_argument("--early-block", type=int, default=2)
    ap.add_argument("--write-layer", type=int, default=None)
    ap.add_argument("--out", default=None,
                    help="output path (default: truth_dictionary_<model>.pt)")
    ap.add_argument("--rev-counterfact",
                    default="c945b082ca08d0a8f3ba227fb78404a09614c36e")
    ap.add_argument("--file-counterfact", default=None)
    a = ap.parse_args()
    wl = a.write_layer if a.write_layer is not None else a.peak + 1
    tag = a.model.split("/")[-1].replace(".", "").replace("-", "_")
    out = a.out or f"truth_dictionary_{tag}.pt"

    dev, _ = T.resolve_device_dtype("auto", "auto")
    dt = torch.float32
    print("[task dictionary_export] measure and save the theorist's dictionary")
    items, pidx, cat_of_pair, cats = load_category_pairs(
        a.k_relations, a.pairs_per_relation, a.seed,
        a.rev_counterfact, a.file_counterfact)
    # keep templates for the bundle
    from datasets import load_dataset
    ds = load_dataset("NeelNanda/counterfact-tracing", split="train",
                      revision=a.rev_counterfact) if not a.file_counterfact \
        else T._open_local(a.file_counterfact)
    templates = {}
    for ex in ds:
        rid = str(ex["relation_id"])
        if rid in cats and rid not in templates:
            templates[rid] = str(ex["relation"])
        if len(templates) == len(cats):
            break
    print(f"[model] {a.model} on {dev} ({dt})")
    tok, model = T.load_model(a.model, dt, dev)
    H_resid, H_attn, H_ffn = A.collect_components(model, tok, items, dev)
    del model
    rel = A.identity_check(H_resid, H_attn, H_ffn)
    print(f"[identity check] median {float(rel.median()):.2e}  (must be ~0)")
    if float(rel.median()) > 1e-3:
        print("  ABORT: decomposition invalid."); return

    H_peak = H_resid[:, a.peak + 1, :].float()
    H_early = H_resid[:, a.early_block + 1, :].float()
    cat_pairs = {c: [pidx[i] for i in range(len(pidx)) if cat_of_pair[i] == c]
                 for c in cats}

    cs, cos_peak, axes_d = axis_cosine_matrix(H_peak, cat_pairs)
    _, cos_early, _ = axis_cosine_matrix(H_early, cat_pairs)
    _, transfer = transfer_matrix(H_peak, cat_pairs, a.folds, a.seed)
    axes = torch.stack([unit(axes_d[c]) for c in cs], 0)
    t_global = unit(T.fit_axis(H_peak, pidx)["v1"])
    Df = torch.stack([unit(H_ffn[it, wl, :].float() - H_ffn[iff, wl, :].float())
                      for it, iff in pidx], 0)
    De = torch.stack([unit(H_early[it] - H_early[iff]) for it, iff in pidx], 0)
    centroids = torch.stack(
        [unit(Df[[i for i in range(len(pidx)) if cat_of_pair[i] == c]].mean(0))
         for c in cs], 0)
    acc_f, nm_f, n95_f, p_f = decoding_with_null(Df, cat_of_pair, a.folds,
                                                 a.seed, a.perm)
    acc_e, nm_e, n95_e, p_e = decoding_with_null(De, cat_of_pair, a.folds,
                                                 a.seed, a.perm)

    meta = dict(model=a.model, peak_block=a.peak, write_layer=wl,
                early_block=a.early_block, dataset="NeelNanda/counterfact-tracing",
                revision=a.rev_counterfact, seed=a.seed, folds=a.folds,
                k_relations=a.k_relations, pairs_per_relation=a.pairs_per_relation,
                templates={c: templates.get(c, "") for c in cs},
                decoding=dict(delta_f=dict(acc=acc_f, null_mean=nm_f,
                                           null_95=n95_f, p=p_f),
                              lexical=dict(acc=acc_e, null_mean=nm_e,
                                           null_95=n95_e, p=p_e)),
                identity_check_median=float(rel.median()))
    bundle = dict(cats=cs, axes=axes, t_global=t_global,
                  write_centroids=centroids, cos_peak=cos_peak,
                  cos_early=cos_early, transfer=transfer, meta=meta)
    torch.save(bundle, out)
    with open(out.replace(".pt", ".json"), "w") as f:
        json.dump(dict(meta, cats=cs,
                       cos_peak=[[round(float(x), 3) for x in r] for r in cos_peak],
                       cos_early=[[round(float(x), 3) for x in r] for r in cos_early],
                       transfer=[[round(float(x), 3) for x in r] for r in transfer]),
                  f, indent=2)
    print(f"\n[saved] {out}  (+ .json human summary)")
    print(f"  cats: {cs}")
    print(f"  axes [K,d] = {tuple(axes.shape)}   t_global [d] = {tuple(t_global.shape)}")
    print(f"  decoding: Delta f {acc_f:.2%} vs lexical {acc_e:.2%} "
          f"(chance {1/len(cs):.0%})")
    print("  load with:  b = torch.load(path); scores = states @ b['axes'].T")


if __name__ == "__main__":
    main()
