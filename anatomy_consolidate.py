# -*- coding: utf-8 -*-
"""
anatomy_consolidate.py  --  Are yesterday's anatomy findings STABLE or noise?

Yesterday (one seed) we saw, per layer, attention and FFN contributions carrying
the truth axis with an apparent alternation (attn leads the middle layers, FFN
holds later). And the expanded FFN space read truth only ~+0.02 above the
residual. Both came from a SINGLE cross-validation seed. Before building on
them, we must know if they SURVIVE re-seeding.

This does NOT add a new experiment. It re-runs the SAME attn/ffn decomposition
across several seeds and reports, per layer, MEAN and STD of attn and ffn AUC,
plus the mean and std of the difference (attn - ffn). A difference is real only
if it clears its own std. The extraction (one forward pass per sentence) is done
ONCE; only the CV re-seeding repeats, so this is cheap.

Reuses anatomy.py (which reuses truth_probe.py). fp32, identity-checked decomposition.
"""
import argparse
import torch
import truth_probe as T
import anatomy as A


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
    ap.add_argument("--dataset", default="builtin",
                    choices=["builtin", "counterfact", "truthfulqa", "mix"])
    ap.add_argument("--max-pairs", type=int, default=250)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seeds", type=int, default=5, help="number of CV seeds to average over")
    ap.add_argument("--rev-counterfact", default=None)
    ap.add_argument("--rev-truthfulqa", default=None)
    ap.add_argument("--file-counterfact", default=None)
    ap.add_argument("--file-truthfulqa", default=None)
    a = ap.parse_args()

    dev, _ = T.resolve_device_dtype("auto", "auto")
    dt = torch.float32
    print("[task anatomy_consolidate] is the attn/ffn picture stable across seeds?")
    print(f"[note] fp32; re-running the SAME decomposition over {a.seeds} CV seeds")
    pairs = list(T.DEFAULT_PAIRS) if a.dataset == "builtin" else \
        T.load_pairs(a.dataset, a.max_pairs, 0, a.rev_counterfact,
                     a.rev_truthfulqa, a.file_counterfact, a.file_truthfulqa)
    items, pidx = T.pairs_to_items(pairs)
    print(f"[data] {len(items)} sentences = {len(pidx)} pairs")
    print(f"[model] {a.model} on {dev} ({dt})")
    tok, model = T.load_model(a.model, dt, dev)

    # extract components ONCE (forward pass is seed-independent)
    H_resid, H_attn, H_ffn = A.collect_components(model, tok, items, dev)
    L = H_attn.shape[1]

    rel = A.identity_check(H_resid, H_attn, H_ffn)
    print(f"\n[identity check] median {float(rel.median()):.2e}  "
          f"(must be ~0 or the decomposition is invalid)")
    if float(rel.median()) > 1e-3:
        print("  ABORT: decomposition invalid, do not read the numbers."); return

    # for each layer, collect attn/ffn AUC across seeds
    attn_by_layer = [[] for _ in range(L)]
    ffn_by_layer = [[] for _ in range(L)]
    resid_by_layer = [[] for _ in range(L)]
    for s in range(a.seeds):
        for Lb in range(L):
            resid_by_layer[Lb].append(A.auc_component(H_resid[:, Lb + 1, :], pidx, a.folds, s))
            attn_by_layer[Lb].append(A.auc_component(H_attn[:, Lb, :], pidx, a.folds, s))
            ffn_by_layer[Lb].append(A.auc_component(H_ffn[:, Lb, :], pidx, a.folds, s))
        print(f"\r[seeds] {s+1}/{a.seeds}", end="", flush=True)
    print()

    def ms(xs):
        t = torch.tensor(xs); return float(t.mean()), float(t.std())

    print(f"\n{'layer':>5} | {'resid':>13} | {'attn':>13} | {'ffn':>13} | {'attn-ffn':>15}")
    print("-" * 70)
    stable_layers = []
    for Lb in range(L):
        rm, rs = ms(resid_by_layer[Lb])
        am, as_ = ms(attn_by_layer[Lb])
        fm, fs = ms(ffn_by_layer[Lb])
        diff = torch.tensor(attn_by_layer[Lb]) - torch.tensor(ffn_by_layer[Lb])
        dm, ds = float(diff.mean()), float(diff.std())
        # a difference is "real" if its magnitude clears its own std (and a small floor)
        real = abs(dm) > max(2 * ds, 0.03)
        flag = ("ATTN" if dm > 0 else "FFN ") if real else "  ~ "
        if real:
            stable_layers.append((Lb, dm, ds, flag.strip()))
        print(f"{Lb:>5} | {rm:.3f}±{rs:.3f} | {am:.3f}±{as_:.3f} | "
              f"{fm:.3f}±{fs:.3f} | {dm:+.3f}±{ds:.3f} {flag}")

    print("\n=== which layers show a STABLE attn vs ffn difference? ===")
    print("    (|attn-ffn| must exceed 2*std and 0.03 to count as real, not noise)")
    if not stable_layers:
        print("  NONE. The apparent attn/ffn alternation does NOT survive re-seeding:")
        print("  it was single-seed noise. The two contributions carry truth equally.")
        print("  -> yesterday's 'attn leads, ffn follows' was an artifact. Honest negative.")
    else:
        attn_wins = [l for l in stable_layers if l[3] == "ATTN"]
        ffn_wins = [l for l in stable_layers if l[3] == "FFN"]
        print(f"  attention dominates (stably) at layers: {[l[0] for l in attn_wins]}")
        print(f"  FFN dominates (stably) at layers       : {[l[0] for l in ffn_wins]}")
        if attn_wins and ffn_wins:
            print("  -> the alternation is REAL: different layers genuinely favor different")
            print("     components. This survives re-seeding and is worth interpreting.")
        else:
            comp = "attention" if attn_wins else "the FFN"
            print(f"  -> only {comp} ever dominates stably; no true alternation, but a real")
            print(f"     and consistent lead for {comp} where it appears.")


if __name__ == "__main__":
    main()
