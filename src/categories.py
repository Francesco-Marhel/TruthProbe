# -*- coding: utf-8 -*-
"""
categories.py  --  Is the truth axis a MIXTURE of category-specific components?
(The pre-SAE contract: if truth directions and FFN write directions carry
category structure beyond lexical surface, a sparse dictionary has something
real to find -- and a registered criterion to be judged by. If they don't,
the SAE chapter dies cheaply, before any training.)

Pre-registered predictions (written before any run):
  P1  cosines between per-category truth axes at the peak: substantially
      below 1 but above 0 -- a shared core (Buerger-style t_G) plus
      category-specific tails.
  P2  transfer matrix: within-category held-out AUC > cross-category AUC.
  P3  the category of a pair is decodable from the FFN's class-signed write
      direction (Delta f at the write layer) above a permutation null.
  SURFACE CONTROL (mandatory): the cosine/decoding structure at the peak must
      DIVERGE from the same structure at an early block; if early layers show
      the same picture, the 'category structure' is lexical surface (targets
      and templates differ by relation), not contextual truth.

Data: CounterFact grouped by relation_id (Wikidata P-codes); top-K relations
by unique pairs, n pairs each. Everything else as in the repo: held-out CV
over pairs, fp32 + identity check, pinned revision, no auto-verdict.

    python categories.py                                   # Qwen2.5-1.5B
    python categories.py --model Qwen/Qwen2.5-3B --peak 16 --write-layer 17
"""
import argparse
import random
import zlib
import torch
import truth_probe as T
import anatomy as A


# =====================================================================
#  pure helpers (unit-testable)
# =====================================================================
def unit(v):
    return v / v.norm().clamp_min(1e-8)


def axis_cosine_matrix(H, cat_pairs):
    """Full-fit axis per category (descriptive) -> KxK SIGNED cos matrix + axes.
    Axes are oriented true=positive within their own category, so the SIGN is
    information: negative = the shared direction is ANTI-aligned to truth in
    the other category (a category-level polarity factor, Buerger-style)."""
    cats = sorted(cat_pairs)
    axes = {c: T.fit_axis(H, cat_pairs[c])["v1"] for c in cats}
    K = len(cats)
    M = torch.zeros(K, K)
    for i, a in enumerate(cats):
        for j, b in enumerate(cats):
            M[i, j] = float(torch.dot(axes[a], axes[b]))
    return cats, M, axes


def transfer_matrix(H, cat_pairs, folds, seed):
    """KxK held-out AUC: diagonal = within-category CV; off-diagonal = axis
    fit on ALL of category A, evaluated on ALL of category B (disjoint)."""
    cats = sorted(cat_pairs)
    K = len(cats)
    M = torch.zeros(K, K)
    for i, a in enumerate(cats):
        pa = cat_pairs[a]
        # within: CV over a's pairs
        aucs = []
        for tr, te in T.kfold_pairs(len(pa), folds, seed):
            ax = T.fit_axis(H, [pa[k] for k in tr])
            I, Y = [], []
            for k in te:
                it, iff = pa[k]; I += [it, iff]; Y += [1, 0]
            aucs.append(T.auc_score(T.project_fields(H[I], ax)["Re"],
                                    torch.tensor(Y)))
        M[i, i] = sum(aucs) / len(aucs)
        ax_full = T.fit_axis(H, pa)
        for j, b in enumerate(cats):
            if i == j:
                continue
            I, Y = [], []
            for it, iff in cat_pairs[b]:
                I += [it, iff]; Y += [1, 0]
            M[i, j] = T.auc_score(T.project_fields(H[I], ax_full)["Re"],
                                  torch.tensor(Y))
    return cats, M


def nearest_centroid_cv(D, labels, folds, seed):
    """Held-out nearest-centroid (cosine) decoding of category from vectors.
    D: [n, d] unit vectors, labels: list of category keys. Returns accuracy."""
    cats = sorted(set(labels))
    idx_by_cat = {c: [i for i, l in enumerate(labels) if l == c] for c in cats}
    # joint folds: fold f takes the f-th slice of every category
    correct, total = 0, 0
    for f in range(folds):
        train, test = [], []
        for c in cats:
            ids = list(idx_by_cat[c])
            random.Random(seed + zlib.crc32(c.encode()) % 1000).shuffle(ids)
            cut = [ids[k::folds] for k in range(folds)]
            test += cut[f]
            train += [i for k in range(folds) if k != f for i in cut[k]]
        cents = {}
        for c in cats:
            tr_c = [i for i in train if labels[i] == c]
            cents[c] = unit(D[tr_c].mean(0))
        C = torch.stack([cents[c] for c in cats], 0)      # [K, d]
        sims = D[test] @ C.T                              # [m, K]
        pred = sims.argmax(dim=1)
        truth = torch.tensor([cats.index(labels[i]) for i in test])
        correct += int((pred == truth).sum()); total += len(test)
    return correct / total


def decoding_with_null(D, labels, folds, seed, perms):
    acc = nearest_centroid_cv(D, labels, folds, seed)
    rng = random.Random(seed)
    null = []
    for b in range(perms):
        lab = list(labels)
        rng.shuffle(lab)
        null.append(nearest_centroid_cv(D, lab, folds, seed))
    nt = torch.tensor(null)
    p = (1 + int((nt >= acc).sum())) / (perms + 1)
    return acc, float(nt.mean()), float(nt.quantile(0.95)), p


# =====================================================================
#  data: CounterFact grouped by relation
# =====================================================================
def load_category_pairs(K, n_per, seed, revision, local_file):
    if local_file:
        ds = T._open_local(local_file)
    else:
        from datasets import load_dataset
        ds = load_dataset("NeelNanda/counterfact-tracing", split="train",
                          revision=revision)
    print(f"  [provenance] NeelNanda/counterfact-tracing @ {revision} ({len(ds)} rows)")
    by_rel, templ = {}, {}
    for ex in ds:
        rid = str(ex["relation_id"])
        prompt = str(ex["prompt"]).strip()
        tt, tf = str(ex["target_true"]), str(ex["target_false"])
        if not prompt or tt.strip() == tf.strip():
            continue
        by_rel.setdefault(rid, {})[(prompt, tt.strip())] = (prompt, tt, tf)
        templ.setdefault(rid, str(ex["relation"]))
    top = sorted(by_rel, key=lambda r: len(by_rel[r]), reverse=True)[:K]
    print(f"  [categories] top-{K} relations by unique pairs:")
    items, cat_of_pair, pidx = [], [], []
    for rid in top:
        rows = list(by_rel[rid].values())
        random.Random(seed).shuffle(rows)
        rows = rows[:n_per]
        print(f"    {rid}: {len(rows)} pairs   template: {templ[rid]!r}")
        for prompt, tt, tf in rows:
            it = len(items); items.append((1, prompt + tt))
            iff = len(items); items.append((0, prompt + tf))
            pidx.append((it, iff)); cat_of_pair.append(rid)
    return items, pidx, cat_of_pair, top


# =====================================================================
#  main
# =====================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
    ap.add_argument("--k-relations", type=int, default=5)
    ap.add_argument("--pairs-per-relation", type=int, default=60)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--perm", type=int, default=100)
    ap.add_argument("--peak", type=int, default=15, help="truth-peak block")
    ap.add_argument("--early-block", type=int, default=2,
                    help="surface-control block (lexical baseline)")
    ap.add_argument("--write-layer", type=int, default=None,
                    help="FFN layer for Delta f (default: peak+1)")
    ap.add_argument("--rev-counterfact",
                    default="c945b082ca08d0a8f3ba227fb78404a09614c36e")
    ap.add_argument("--file-counterfact", default=None)
    a = ap.parse_args()
    wl = a.write_layer if a.write_layer is not None else a.peak + 1

    dev, _ = T.resolve_device_dtype("auto", "auto")
    dt = torch.float32
    print("[task categories] is the truth axis a mixture of per-category components?")
    print(f"[note] axes at block {a.peak}; surface control at block {a.early_block}; "
          f"FFN writes at layer {wl}")
    items, pidx, cat_of_pair, cats = load_category_pairs(
        a.k_relations, a.pairs_per_relation, a.seed,
        a.rev_counterfact, a.file_counterfact)
    print(f"[data] {len(items)} sentences = {len(pidx)} pairs, {len(cats)} categories")
    print(f"[model] {a.model} on {dev} ({dt})")
    tok, model = T.load_model(a.model, dt, dev)

    H_resid, H_attn, H_ffn = A.collect_components(model, tok, items, dev)
    del model
    rel = A.identity_check(H_resid, H_attn, H_ffn)
    print(f"[identity check] median {float(rel.median()):.2e}  (must be ~0)")
    if float(rel.median()) > 1e-3:
        print("  ABORT: decomposition invalid."); return

    cat_pairs = {c: [pidx[i] for i in range(len(pidx)) if cat_of_pair[i] == c]
                 for c in cats}
    H_peak = H_resid[:, a.peak + 1, :].float()
    H_early = H_resid[:, a.early_block + 1, :].float()

    # ---------- (P1) axis cosines, peak vs early (surface control) ----------
    for name, H in [("PEAK", H_peak), (f"EARLY (block {a.early_block})", H_early)]:
        cs, M, _ = axis_cosine_matrix(H, cat_pairs)
        print(f"\n=== SIGNED cos between per-category truth axes @ {name} ===")
        print("       " + "".join(f"{c:>8}" for c in cs))
        for i, c in enumerate(cs):
            print(f"{c:>7}" + "".join(f"{float(M[i,j]):>8.2f}" for j in range(len(cs))))
        off = [float(M[i, j]) for i in range(len(cs)) for j in range(len(cs)) if i != j]
        neg = [x for x in off if x < -0.05]
        print(f"  off-diagonal mean {sum(off)/len(off):+.2f}   "
              f"negative entries: {len(neg)}//{len(off)} (anti-aligned pairs)")

    # ---------- (P2) transfer matrix at the peak ----------
    cs, TM = transfer_matrix(H_peak, cat_pairs, a.folds, a.seed)
    print("\n=== held-out AUC transfer @ PEAK (row axis -> column data; "
          "diagonal = within-CV) ===")
    print("       " + "".join(f"{c:>8}" for c in cs))
    for i, c in enumerate(cs):
        print(f"{c:>7}" + "".join(f"{float(TM[i,j]):>8.3f}" for j in range(len(cs))))
    diag = [float(TM[i, i]) for i in range(len(cs))]
    off = [float(TM[i, j]) for i in range(len(cs)) for j in range(len(cs)) if i != j]
    print(f"  within mean {sum(diag)/len(diag):.3f}   cross mean {sum(off)/len(off):.3f}")

    # ---------- (P3) category decodability of FFN write directions ----------
    Df = torch.stack([unit(H_ffn[it, wl, :].float() - H_ffn[iff, wl, :].float())
                      for it, iff in pidx], 0)
    De = torch.stack([unit(H_early[it] - H_early[iff]) for it, iff in pidx], 0)
    print(f"\n=== category decoding from class-signed directions "
          f"(nearest centroid, {a.folds}-fold, {a.perm} perms) ===")
    for name, D in [(f"FFN write Delta f @ layer {wl}", Df),
                    (f"EARLY residual Delta h @ block {a.early_block} "
                     "(lexical baseline)", De)]:
        acc, nmean, n95, p = decoding_with_null(D, cat_of_pair, a.folds,
                                                a.seed, a.perm)
        print(f"  {name}:")
        print(f"    accuracy {acc:.2%}   null mean {nmean:.2%}  null 95pct {n95:.2%}"
              f"   p={p:.4f}   (chance = {1/len(cats):.2%})")

    print("\n=== reading guide (no auto-verdict; predictions registered) ===")
    print("  P1: peak off-diagonal cosines well below 1 but above 0 = shared core")
    print("      + category tails. SURFACE CONTROL: if the EARLY matrix shows the")
    print("      same structure, the tails are lexical, not contextual.")
    print("  P2: within > cross at the peak = category-specific truth components.")
    print("  P3: Delta f decodes category above null = the FFN writes truth along")
    print("      category-dependent directions -- the SAE's target exists. But if")
    print("      the EARLY lexical baseline decodes just as well, target-token")
    print("      identity may explain it: compare the two accuracies, not P3 alone.")
    print("  SAE contract: train only if P1-P3 hold beyond the surface control;")
    print("  registered criterion: SAE features must align to per-category axes")
    print("  above a matched null. Either outcome is content.")


if __name__ == "__main__":
    main()
