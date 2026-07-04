# -*- coding: utf-8 -*-
"""
axis_provenance.py (v2) -- Is the b/b+1 relational pattern genuine alignment
or partly circular (f_b is inside the very state v1@b is fit on)?

Two rulers per fit block b, from the SAME cached tensors:
  post : v1 fit on the block-b OUTPUT  h_{b+1} = h_b + a_b + f_b   (contains f_b)
  pre  : v1 fit on the PRE-FFN state   h_b + a_b                   (f_b excluded)

Pre-registered discrimination for the 'pro at own block' term:
  GENUINE  : f_b stays pro-aligned under the PRE ruler too
             (the FFN writes along the direction the stream is already
              building -> relational law complete: builds-forward,
              rotates-away-from-behind).
  CIRCULAR : f_b's alignment collapses to ~0 under the PRE ruler
             (the pro term was manufactured by fitting the ruler on a state
              that contains f_b; the clean law is only 'anti-previous').
The 'anti at b+1' term is not at risk of this circularity (f_{b+1} is not in
the block-b states) and should appear under both rulers.

NO automatic verdict: the transition rule misled twice tonight. The script
prints the matrices and a per-ruler diagonal summary; the human reads.

    python axis_provenance.py                       # Qwen2.5-1.5B
    python axis_provenance.py --model Qwen/Qwen2.5-3B --peak 16 \
        --scan-start 13 --scan-end 21
"""
import argparse
import torch
import truth_probe as T
import anatomy as A
from flip_consolidate import axis_per_fold, gap_on_axis


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
    ap.add_argument("--dataset", default="counterfact",
                    choices=["builtin", "counterfact", "truthfulqa", "mix"])
    ap.add_argument("--max-pairs", type=int, default=250)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--peak", type=int, default=15, help="truth-peak block p (for markers only)")
    ap.add_argument("--axis-offsets", default="-4,-2,0,2")
    ap.add_argument("--component", default="ffn", choices=["ffn", "attn"],
                    help="which contribution to project on the rulers")
    ap.add_argument("--scan-start", type=int, default=12)
    ap.add_argument("--scan-end", type=int, default=20)
    ap.add_argument("--rev-counterfact", default="c945b082ca08d0a8f3ba227fb78404a09614c36e")
    ap.add_argument("--rev-truthfulqa", default="741b8276f2d1982aa3d5b832d3ee81ed3b896490")
    ap.add_argument("--file-counterfact", default=None)
    ap.add_argument("--file-truthfulqa", default=None)
    a = ap.parse_args()

    dev, _ = T.resolve_device_dtype("auto", "auto")
    dt = torch.float32
    offsets = [int(x) for x in a.axis_offsets.split(",")]
    axis_blocks = [a.peak + o for o in offsets]
    print("[task axis_provenance v2] genuine alignment or circular ruler?")
    print(f"[note] rulers at blocks {axis_blocks}; POST = block output (contains f_b),")
    print(f"       PRE = h_b + a_b (f_b excluded). FFN gap scanned at {a.scan_start}-{a.scan_end}.")
    pairs = list(T.DEFAULT_PAIRS) if a.dataset == "builtin" else \
        T.load_pairs(a.dataset, a.max_pairs, a.seed, a.rev_counterfact,
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

    def ruler_states(b, kind):
        if kind == "post":
            return H_resid[:, b + 1, :]
        return H_resid[:, b, :] + H_attn[:, b, :]          # pre-FFN state of block b

    for kind in ("post", "pre"):
        print(f"\n================  RULER = {kind.upper()}  ================")
        cols = {b: axis_per_fold(ruler_states(b, kind), pidx, a.folds, a.seed)
                for b in axis_blocks}
        print(f"{'layer':>5} |" + "".join(f"  v1@{b:>3}" for b in axis_blocks))
        print("-" * (8 + 8 * len(axis_blocks)))
        dmat = {}
        for Lb in scan:
            Hc = H_ffn if a.component == "ffn" else H_attn
            row = [gap_on_axis(Hc[:, Lb, :], pidx, cols[b])[1] for b in axis_blocks]
            dmat[Lb] = row
            mark = "<- p" if Lb == a.peak else ("<- p+1" if Lb == a.peak + 1 else "")
            print(f"{Lb:>5} |" + "".join(f" {v:>+6.2f}" for v in row) + f"  {mark}")
        comp = a.component.upper()
        print(f"\n  diagonal summary ({kind}): per ruler b, the {comp}'s d' AT b and AT b+1")
        print(f"  {'ruler b':>8} | {'d(b)':>9} | {'d(b+1)':>10}")
        for j, b in enumerate(axis_blocks):
            db = dmat[b][j] if b in dmat else float('nan')
            db1 = dmat[b + 1][j] if (b + 1) in dmat else float('nan')
            print(f"  {b:>8} | {db:>+9.2f} | {db1:>+10.2f}")

    print("\n=== reading guide (no auto-verdict; pre-registered predictions) ===")
    print("  POST diagonal: pro at b, anti at b+1 (already seen; possibly part-circular).")
    print("  PRE  diagonal decides:")
    print("    d(b) stays clearly positive  -> GENUINE: the FFN writes along the")
    print("      direction the stream is already building. Relational law complete.")
    print("    d(b) collapses toward zero   -> CIRCULAR: the pro term was the ruler")
    print("      measuring its own ingredient. The clean law is only 'anti-previous'")
    print("      (the FFN as rotation engine).")
    print("    d(b+1) should stay negative under BOTH rulers (not at risk).")


if __name__ == "__main__":
    main()
