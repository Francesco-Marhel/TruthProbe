# -*- coding: utf-8 -*-
"""
crea_dizionario.py  --  Stand-alone, all-in-one builder of the measured "truth
dictionary". ONE self-contained file: it depends on nothing else in this folder
(no truth_probe / anatomy / categories imports). Every step is inlined verbatim
from the tested sources, so the numbers are identical -- but the data no longer
travels file-to-file (the "passa-parola" that confused people). One file in,
one clean bundle out.

WHAT IT DOES, per model:
    load CounterFact grouped by relation (top-K categories)
    load the model (safetensors, fp32)
    split the residual stream into attn + FFN contributions (forward hooks)
    identity check  attn+ffn == residual delta   (median ~0 or ABORT)      <-- gate
    per-category truth axes at the PEAK block  ->  SIGNED cosine matrix
    the same at an EARLY block                 ->  surface control
    held-out CV transfer matrix (within vs cross category)
    FFN class-signed write directions (Delta f) -> per-category centroids
    category decoding of Delta f vs a lexical baseline, with a permutation null
    SAVE the bundle exactly like dictionary_export.py:  <out>/truth_dictionary_<tag>.pt  (+ .json)

VALIDATION PROTOCOL kept intact: fp32 identity gate, k-fold held-out over pairs
(never splitting a pair), permutation nulls, and the mandatory EARLY surface
control. SECURITY kept intact: HuggingFace model loaded with safetensors only and
trust_remote_code=False; the dataset is read as Parquet (no remote code) at a
PINNED revision.

GPU: defaults to cuda in fp32. fp32 is REQUIRED (the attn+ffn==delta identity
suffers catastrophic cancellation in bf16). A 3B model in fp32 is ~12 GB, i.e.
it does not fully fit a 12 GB card: when the pure-GPU load OOMs, the script
automatically switches to device_map='auto' (keeps as much as possible on the
GPU, offloads the rest to CPU, fp32 preserved). Small models (<=1.5B) run fully
on the GPU. Use --no-offload to force a plain CPU fallback instead.

    python crea_dizionario.py                       # every model in the registry
    python crea_dizionario.py --models Qwen/Qwen2.5-3B
    python crea_dizionario.py --verify              # also check vs the canonical K=8 archive
    python crea_dizionario.py --models <hf/name> --peak 20 --write-layer 21   # a new model
    python crea_dizionario.py --list-relations      # list candidate categories to a .txt, then exit
    python crea_dizionario.py --all-layers          # also save per-layer cosine matrices (timelapse source)
    python crea_dizionario.py --flip-layers         # also save the FFN pro->anti-truth flip per block

To EXPAND MODELS or CATEGORIES, edit the clearly marked USER CONFIG block below.
"""
import os
import json
import argparse
import random
import zlib
import torch


# =====================================================================
#  ================   USER CONFIG  --  EDIT HERE   ================
# =====================================================================
# (1) MODELS registry.  name -> the block where the truth axis PEAKS and the
#     layer where the FFN WRITES (= peak + 1), plus the shallow EARLY control
#     block. The peak is MODEL-SPECIFIC and must be MEASURED (truth-axis signal
#     scan per layer), never guessed, before you add a new model here.
MODELS = {
    "Qwen/Qwen2.5-3B":         dict(peak=16, write_layer=17, early_block=2),
    "meta-llama/Llama-3.2-3B": dict(peak=9,  write_layer=10, early_block=2),
    # already-characterized smaller siblings (uncomment to include them):
    # "Qwen/Qwen2.5-1.5B":       dict(peak=15, write_layer=16, early_block=2),
    # "meta-llama/Llama-3.2-1B": dict(peak=7,  write_layer=8,  early_block=2),
    # ---- ADD A NEW MODEL HERE ----
    # "<hf/org/model-name>":    dict(peak=<measured>, write_layer=<peak+1>, early_block=2),
}

# (2) CATEGORIES.  Edit these three lines to change/expand the categories.
K_RELATIONS = 8             # number of categories = top-K CounterFact relations by pair count
PAIRS_PER_RELATION = 60     # pairs sampled per relation (balanced K x N design)
RELATION_WHITELIST = None   # None = automatic top-K.  To FORCE an exact set (add a
                            # category the auto top-K would miss, or pin a set for
                            # reproducibility) put P-codes here, e.g.:
                            #   RELATION_WHITELIST = ["P103","P1412","P176","P27",
                            #                         "P30","P37","P413","P495"]

# (3) DATASET.  Parquet-only, no trust_remote_code; the pinned revision makes the
#     category selection reproducible. Point --file-counterfact at a local parquet
#     to run fully offline.
DATASET_REPO = "NeelNanda/counterfact-tracing"
DATASET_REVISION = "c945b082ca08d0a8f3ba227fb78404a09614c36e"
EARLY_BLOCK = 2             # default early/surface-control block for new models
# =====================================================================
#  ================   END USER CONFIG   ================
# =====================================================================


DTYPE_MAP = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}


# =====================================================================
#  geometry + metrics  (pure; identical to the paper's truth_probe.py)
# =====================================================================
def unit(v):
    return v / v.norm().clamp_min(1e-8)


def auc_score(s, y):
    s = s.float(); y = y.long()
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    gt = (pos[:, None] > neg[None, :]).float().mean()
    eq = (pos[:, None] == neg[None, :]).float().mean()
    return float(gt + 0.5 * eq)


def ang_diff(a, b):
    d = a - b
    return torch.atan2(torch.sin(d), torch.cos(d))


def robust_ms(c):
    med = c.median()
    scale = (1.4826 * (c - med).abs().median()).clamp_min(1e-8)
    return float(med), float(scale)


def kfold_pairs(n_pairs, k, seed=0):
    idx = list(range(n_pairs)); random.Random(seed).shuffle(idx)
    folds = [idx[i::k] for i in range(k)]
    for i in range(k):
        test = set(folds[i]); train = [p for p in idx if p not in test]
        yield train, sorted(folds[i])


def fit_axis(Hl, pidx, orient=None):
    """Unsupervised truth axis from the SVD of intra-pair (true - false)
    differences, oriented true = positive. Identical to truth_probe.fit_axis."""
    if orient is None:
        orient = [1] * len(pidx)
    Hl = Hl.float()
    t_idx, f_idx = [], []
    for (it, iff), o in zip(pidx, orient):
        (t_idx if o >= 0 else f_idx).append(it)
        (f_idx if o >= 0 else t_idx).append(iff)
    D = Hl[t_idx] - Hl[f_idx]
    _, _, Vh = torch.linalg.svd(D, full_matrices=False)
    v1 = Vh[0].clone()
    v2 = Vh[1] - torch.dot(Vh[1], v1) * v1
    v2 = v2 / v2.norm().clamp_min(1e-8)
    proj_t, proj_f = Hl[t_idx] @ v1, Hl[f_idx] @ v1
    if proj_t.mean() < proj_f.mean():
        v1 = -v1; proj_t, proj_f = -proj_t, -proj_f
    states = torch.cat([Hl[t_idx], Hl[f_idx]], 0)
    c1, c2 = states @ v1, states @ v2
    med1, s1 = robust_ms(c1); med2, s2 = robust_ms(c2)
    Re = (c1 - med1) / s1; Im = (c2 - med2) / s2
    r = float(torch.sqrt(Re ** 2 + Im ** 2).median().clamp_min(1e-8))
    Re_t = (proj_t - med1) / s1; Im_t = ((Hl[t_idx] @ v2) - med2) / s2
    th_true = float(torch.atan2(torch.sin(torch.atan2(Im_t, Re_t)).mean(),
                                torch.cos(torch.atan2(Im_t, Re_t)).mean()))
    return dict(v1=v1, v2=v2, med1=med1, s1=s1, med2=med2, s2=s2, r=r, th_true=th_true)


def project_fields(Hl, ax):
    Hl = Hl.float()
    Re = (Hl @ ax["v1"] - ax["med1"]) / ax["s1"]
    Im = (Hl @ ax["v2"] - ax["med2"]) / ax["s2"]
    b = torch.sigmoid(Re); m = torch.sqrt(Re ** 2 + Im ** 2)
    theta = torch.atan2(Im, Re)
    risk = (0.5 - b) * torch.tanh(m / max(ax["r"], 1e-8))
    pdev = ang_diff(theta, ax["th_true"]).abs()
    return dict(Re=Re, Im=Im, b=b, m=m, theta=theta, risk=risk, phase_dev=pdev)


# =====================================================================
#  model loading (safetensors, GPU + automatic fp32 CPU-offload) + extraction
# =====================================================================
def find_decoder_layers(model):
    """Locate the decoder-block ModuleList across common decoder-only stacks.
    The decomposition needs, per block, a `self_attn` and an `mlp` submodule
    (Llama / Qwen2 / Mistral / Gemma / Phi families)."""
    candidates = []
    inner = getattr(model, "model", model)
    if hasattr(inner, "layers"):
        candidates.append(inner.layers)
    tr = getattr(model, "transformer", None)
    if tr is not None and hasattr(tr, "h"):
        candidates.append(tr.h)
    for layers in candidates:
        if layers is not None and len(layers) > 0 \
           and hasattr(layers[0], "self_attn") and hasattr(layers[0], "mlp"):
            return layers
    raise RuntimeError(
        "Could not find decoder layers exposing .self_attn and .mlp. This script "
        "supports the Llama/Qwen2/Mistral/Gemma-style decoder-only stack. Add your "
        "architecture's module path in find_decoder_layers().")


def load_model(name, dtype, device, offload=True):
    """Security: safetensors only, trust_remote_code=False. Returns
    (tokenizer, model, input_device). On a GPU too small for fp32 weights, falls
    back to device_map='auto' (GPU + CPU offload, fp32 kept) or plain CPU."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name, trust_remote_code=False)
    common = dict(use_safetensors=True, trust_remote_code=False, low_cpu_mem_usage=True)

    def _load(**extra):
        try:                                   # recent transformers: dtype=
            return AutoModelForCausalLM.from_pretrained(name, dtype=dtype, **common, **extra)
        except TypeError:                      # older transformers: torch_dtype=
            return AutoModelForCausalLM.from_pretrained(name, torch_dtype=dtype, **common, **extra)

    if device == "cuda":
        try:
            model = _load(); model.to("cuda")
        except RuntimeError as e:              # torch.cuda.OutOfMemoryError is a RuntimeError
            if "out of memory" not in str(e).lower():
                raise
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if offload:
                try:
                    print("  [vram] fp32 weights do not fit fully on the GPU; using "
                          "device_map='auto' (GPU + CPU offload, fp32 preserved, slower).")
                    model = _load(device_map="auto")
                except Exception as e2:
                    print(f"  [vram] offload unavailable ({type(e2).__name__}); CPU fp32 fallback.")
                    model = _load(); model.to("cpu")
            else:
                print("  [vram] OOM on GPU; CPU fp32 fallback (slower).")
                model = _load(); model.to("cpu")
    else:
        model = _load(); model.to(device)

    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    in_dev = model.get_input_embeddings().weight.device      # robust for device_map too
    return tok, model, in_dev


def collect_components(model, tok, items, in_dev):
    """One forward per sentence. Returns last-token states:
       H_resid [N, L+1, d]  residual stream (embedding + each block output)
       H_attn  [N, L,   d]  per block: the vector attention adds
       H_ffn   [N, L,   d]  per block: the vector the FFN adds"""
    layers = find_decoder_layers(model)
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
            ids = tok(txt, return_tensors="pt").to(in_dev)
            with torch.no_grad():
                out = model(**ids, output_hidden_states=True)
            hs = out.hidden_states
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
    """attn[i] + ffn[i] must equal resid[i+1] - resid[i]. Returns per-(sentence,
    layer) relative error; the MEDIAN must be ~0 or the decomposition is invalid."""
    L = H_attn.shape[1]
    delta = H_resid[:, 1:L + 1, :] - H_resid[:, 0:L, :]
    recon = H_attn + H_ffn
    abs_err = (recon - delta).norm(dim=-1)
    scale = delta.norm(dim=-1).clamp_min(1e-6)
    return abs_err / scale


# =====================================================================
#  data: CounterFact grouped by relation (Parquet-only, pinned revision)
# =====================================================================
def open_local(local_file):
    from datasets import load_dataset
    ext = os.path.splitext(local_file)[1].lower().lstrip(".")
    fmt = {"parquet": "parquet", "json": "json", "jsonl": "json", "csv": "csv"}.get(ext)
    if fmt is None:
        raise ValueError(f"unsupported local extension: {ext}")
    return load_dataset(fmt, data_files=local_file, split="train")


def load_category_pairs(k, n_per, seed, revision, local_file, whitelist=None):
    if local_file:
        ds = open_local(local_file)
    else:
        from datasets import load_dataset
        ds = load_dataset(DATASET_REPO, split="train", revision=revision)   # no trust_remote_code
    print(f"  [provenance] {DATASET_REPO} @ {revision} ({len(ds)} rows)")
    by_rel, templ = {}, {}
    for ex in ds:
        rid = str(ex["relation_id"])
        prompt = str(ex["prompt"]).strip()
        tt, tf = str(ex["target_true"]), str(ex["target_false"])
        if not prompt or tt.strip() == tf.strip():
            continue
        by_rel.setdefault(rid, {})[(prompt, tt.strip())] = (prompt, tt, tf)
        templ.setdefault(rid, str(ex["relation"]))
    if whitelist:
        top = [r for r in whitelist if r in by_rel]
        missing = [r for r in whitelist if r not in by_rel]
        if missing:
            print(f"  [categories] WARNING: whitelisted relations not present: {missing}")
        print(f"  [categories] whitelist ({len(top)} relations):")
    else:
        top = sorted(by_rel, key=lambda r: len(by_rel[r]), reverse=True)[:k]
        print(f"  [categories] top-{k} relations by unique pairs:")
    items, cat_of_pair, pidx, templates = [], [], [], {}
    for rid in top:
        rows = list(by_rel[rid].values())
        random.Random(seed).shuffle(rows)
        rows = rows[:n_per]
        templates[rid] = templ.get(rid, "")
        print(f"    {rid}: {len(rows)} pairs   template: {templ[rid]!r}")
        for prompt, tt, tf in rows:
            it = len(items); items.append((1, prompt + tt))
            iff = len(items); items.append((0, prompt + tf))
            pidx.append((it, iff)); cat_of_pair.append(rid)
    return items, pidx, cat_of_pair, top, templates


# =====================================================================
#  cosine matrix / transfer / category decoding  (identical to categories.py)
# =====================================================================
def axis_cosine_matrix(H, cat_pairs):
    """Full-fit truth axis per category -> KxK SIGNED cosine matrix + axes."""
    cats = sorted(cat_pairs)
    axes = {c: fit_axis(H, cat_pairs[c])["v1"] for c in cats}
    K = len(cats)
    M = torch.zeros(K, K)
    for i, a in enumerate(cats):
        for j, b in enumerate(cats):
            M[i, j] = float(torch.dot(axes[a], axes[b]))
    return cats, M, axes


def transfer_matrix(H, cat_pairs, folds, seed):
    """Held-out AUC: diagonal = within-category CV; off-diagonal = axis fit on
    ALL of category A, evaluated on ALL of category B."""
    cats = sorted(cat_pairs)
    K = len(cats)
    M = torch.zeros(K, K)
    for i, a in enumerate(cats):
        pa = cat_pairs[a]
        aucs = []
        for tr, te in kfold_pairs(len(pa), folds, seed):
            ax = fit_axis(H, [pa[k] for k in tr])
            I, Y = [], []
            for k in te:
                it, iff = pa[k]; I += [it, iff]; Y += [1, 0]
            aucs.append(auc_score(project_fields(H[I], ax)["Re"], torch.tensor(Y)))
        M[i, i] = sum(aucs) / len(aucs)
        ax_full = fit_axis(H, pa)
        for j, b in enumerate(cats):
            if i == j:
                continue
            I, Y = [], []
            for it, iff in cat_pairs[b]:
                I += [it, iff]; Y += [1, 0]
            M[i, j] = auc_score(project_fields(H[I], ax_full)["Re"], torch.tensor(Y))
    return cats, M


def nearest_centroid_cv(D, labels, folds, seed):
    """Held-out nearest-centroid (cosine) decoding of category from unit vectors."""
    cats = sorted(set(labels))
    idx_by_cat = {c: [i for i, l in enumerate(labels) if l == c] for c in cats}
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
        C = torch.stack([cents[c] for c in cats], 0)
        sims = D[test] @ C.T
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
#  canonical K=8 archive (July 5 2026) -- OPTIONAL reproducibility self-check
# =====================================================================
CANON_CATS = ["P103", "P1412", "P176", "P27", "P30", "P37", "P413", "P495"]
REFERENCE = {
    "Qwen/Qwen2.5-3B": dict(cats=CANON_CATS, cos_peak=[
        [1.00, 0.36, 0.13, 0.28, -0.10, 0.27, -0.02, 0.11],
        [0.36, 1.00, 0.11, 0.38, -0.13, 0.76, -0.06, 0.09],
        [0.13, 0.11, 1.00, 0.25, 0.07, 0.00, 0.00, 0.03],
        [0.28, 0.38, 0.25, 1.00, -0.20, 0.28, -0.04, 0.16],
        [-0.10, -0.13, 0.07, -0.20, 1.00, -0.07, -0.09, 0.05],
        [0.27, 0.76, 0.00, 0.28, -0.07, 1.00, -0.06, 0.09],
        [-0.02, -0.06, 0.00, -0.04, -0.09, -0.06, 1.00, -0.01],
        [0.11, 0.09, 0.03, 0.16, 0.05, 0.09, -0.01, 1.00]], cos_early=[
        [1.00, 0.31, 0.04, -0.19, -0.08, -0.26, -0.05, 0.09],
        [0.31, 1.00, -0.02, -0.24, -0.03, -0.90, -0.09, 0.08],
        [0.04, -0.02, 1.00, -0.03, 0.01, 0.03, 0.03, 0.01],
        [-0.19, -0.24, -0.03, 1.00, 0.23, 0.23, 0.03, 0.27],
        [-0.08, -0.03, 0.01, 0.23, 1.00, 0.04, -0.10, 0.13],
        [-0.26, -0.90, 0.03, 0.23, 0.04, 1.00, 0.08, -0.09],
        [-0.05, -0.09, 0.03, 0.03, -0.10, 0.08, 1.00, 0.01],
        [0.09, 0.08, 0.01, 0.27, 0.13, -0.09, 0.01, 1.00]]),
    "meta-llama/Llama-3.2-3B": dict(cats=CANON_CATS, cos_peak=[
        [1.00, 0.69, 0.40, 0.66, -0.14, 0.56, 0.00, 0.51],
        [0.69, 1.00, 0.38, 0.62, -0.15, 0.62, 0.01, 0.52],
        [0.40, 0.38, 1.00, 0.51, -0.01, 0.40, 0.06, 0.51],
        [0.66, 0.62, 0.51, 1.00, -0.09, 0.55, 0.06, 0.75],
        [-0.14, -0.15, -0.01, -0.09, 1.00, -0.09, -0.06, -0.03],
        [0.56, 0.62, 0.40, 0.55, -0.09, 1.00, 0.03, 0.52],
        [0.00, 0.01, 0.06, 0.06, -0.06, 0.03, 1.00, 0.06],
        [0.51, 0.52, 0.51, 0.75, -0.03, 0.52, 0.06, 1.00]], cos_early=[
        [1.00, 0.08, 0.05, -0.05, -0.06, -0.18, -0.03, 0.03],
        [0.08, 1.00, 0.04, -0.16, -0.06, -0.86, -0.03, 0.08],
        [0.05, 0.04, 1.00, 0.10, -0.05, -0.04, 0.00, 0.08],
        [-0.05, -0.16, 0.10, 1.00, 0.17, 0.21, 0.04, 0.11],
        [-0.06, -0.06, -0.05, 0.17, 1.00, 0.08, -0.01, 0.07],
        [-0.18, -0.86, -0.04, 0.21, 0.08, 1.00, 0.03, -0.11],
        [-0.03, -0.03, 0.00, 0.04, -0.01, 0.03, 1.00, -0.01],
        [0.03, 0.08, 0.08, 0.11, 0.07, -0.11, -0.01, 1.00]]),
}


def verify_against_reference(model_name, bundle, tol):
    ref = REFERENCE.get(model_name)
    if ref is None:
        print(f"  [verify] no archived reference for {model_name}; skipping."); return
    if list(bundle["cats"]) != ref["cats"]:
        print("  [verify] category set differs from the K=8 archive; skipping."); return
    import numpy as np
    print(f"  [verify] {model_name} vs canonical K=8 archive (tol {tol}; the archive is")
    print("           rounded to 2 decimals, so ~0.005/cell plus small fp32 drift is normal):")
    for key in ("cos_peak", "cos_early"):
        M = bundle[key].detach().cpu().numpy()
        R = np.asarray(ref[key], dtype=float)
        off = ~np.eye(len(R), dtype=bool)
        max_off = float(np.abs(M - R)[off].max())
        print(f"    {key:9s}: max off-diagonal |fresh - archived| = {max_off:.3f}  "
              f"-> {'MATCH' if max_off <= tol else 'DIVERGES'}")


# =====================================================================
#  the FFN pro->anti-truth flip, per block (for --flip-layers)
# =====================================================================
def flip_curves(H_resid, H_attn, H_ffn, pidx, peak, folds, seed):
    """Per-block class gap of the FFN and attention CONTRIBUTIONS, read on the
    FIXED truth axis fit at the peak block, held out over pairs. The FFN gap is
    pro-truth at the peak and flips to ANTI-truth at peak+1 -- the measured flip
    (same quantity as flip_consolidate.py). Returns per-block lists + landmarks."""
    Hax = H_resid[:, peak + 1, :]
    nB = H_ffn.shape[1]
    gap_f = [[] for _ in range(nB)]; dp_f = [[] for _ in range(nB)]
    gap_a = [[] for _ in range(nB)]; dp_a = [[] for _ in range(nB)]
    for tr, te in kfold_pairs(len(pidx), folds, seed):
        v1 = fit_axis(Hax, [pidx[p] for p in tr])["v1"]     # axis oriented true = positive at peak
        tt = [pidx[p][0] for p in te]; ff = [pidx[p][1] for p in te]
        for L in range(nB):
            for Hc, gl, dl in ((H_ffn, gap_f, dp_f), (H_attn, gap_a, dp_a)):
                pt = Hc[tt, L, :].float() @ v1
                pf = Hc[ff, L, :].float() @ v1
                gap = float(pt.mean() - pf.mean())
                pooled = float(torch.sqrt((pt.var() + pf.var()) / 2).clamp_min(1e-8))
                gl[L].append(gap); dl[L].append(gap / pooled)
    m = lambda xs: [sum(c) / len(c) for c in xs]
    return dict(gap_ffn=m(gap_f), dprime_ffn=m(dp_f),
                gap_attn=m(gap_a), dprime_attn=m(dp_a),
                axis_block=peak, flip_block=peak + 1, n_blocks=nB)


# =====================================================================
#  build + export one dictionary bundle (same output as dictionary_export.py)
# =====================================================================
def build_and_export(model_name, peak, write_layer, early_block, a):
    dev = ("cuda" if torch.cuda.is_available() else "cpu") if a.device == "auto" else a.device
    dt = DTYPE_MAP[a.dtype]
    print("\n" + "=" * 90)
    print(f"[dictionary] {model_name}   peak {peak}   FFN write {write_layer}   early {early_block}")
    print("=" * 90)
    tag = model_name.split("/")[-1].replace(".", "").replace("-", "_")
    out = os.path.join(a.out_dir, f"truth_dictionary_{tag}.pt")
    if os.path.exists(out) and not a.force:   # FAIL FAST: check BEFORE the expensive extraction
        print(f"[skip] {out} already exists -- NOT recomputing (it would not be saved,")
        print("       so the extraction would be wasted). Pass --force to overwrite it,")
        print("       or --out-dir <folder> to write elsewhere.")
        return None
    if dt is not torch.float32:
        print("  [warn] dtype != float32: the attn+ffn==residual identity needs fp32; the")
        print("         identity gate will likely ABORT. Keep float32 unless you know why.")

    items, pidx, cat_of_pair, cats, templates = load_category_pairs(
        a.k_relations, a.pairs_per_relation, a.seed, a.rev_counterfact,
        a.file_counterfact, whitelist=RELATION_WHITELIST)
    print(f"[data] {len(items)} sentences = {len(pidx)} pairs, {len(cats)} categories")
    print(f"[model] {model_name} on {dev} ({dt})")
    tok, model, in_dev = load_model(model_name, dt, dev, offload=not a.no_offload)

    H_resid, H_attn, H_ffn = collect_components(model, tok, items, in_dev)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    rel = identity_check(H_resid, H_attn, H_ffn)
    med = float(rel.median())
    print(f"[identity check] median {med:.2e}  (must be ~0)")
    if med > 1e-3:
        print("  ABORT: decomposition invalid -> not exporting this model.")
        return None

    H_peak = H_resid[:, peak + 1, :].float()
    H_early = H_resid[:, early_block + 1, :].float()
    cat_pairs = {c: [pidx[i] for i in range(len(pidx)) if cat_of_pair[i] == c] for c in cats}

    cs, cos_peak, axes_d = axis_cosine_matrix(H_peak, cat_pairs)
    _, cos_early, _ = axis_cosine_matrix(H_early, cat_pairs)
    _, transfer = transfer_matrix(H_peak, cat_pairs, a.folds, a.seed)
    axes = torch.stack([unit(axes_d[c]) for c in cs], 0)
    t_global = unit(fit_axis(H_peak, pidx)["v1"])
    Df = torch.stack([unit(H_ffn[it, write_layer, :].float() - H_ffn[iff, write_layer, :].float())
                      for it, iff in pidx], 0)
    De = torch.stack([unit(H_early[it] - H_early[iff]) for it, iff in pidx], 0)
    centroids = torch.stack(
        [unit(Df[[i for i in range(len(pidx)) if cat_of_pair[i] == c]].mean(0)) for c in cs], 0)
    acc_f, nm_f, n95_f, p_f = decoding_with_null(Df, cat_of_pair, a.folds, a.seed, a.perm)
    acc_e, nm_e, n95_e, p_e = decoding_with_null(De, cat_of_pair, a.folds, a.seed, a.perm)

    meta = dict(model=model_name, peak_block=peak, write_layer=write_layer,
                early_block=early_block, dataset=DATASET_REPO,
                revision=a.rev_counterfact, seed=a.seed, folds=a.folds,
                k_relations=len(cs), pairs_per_relation=a.pairs_per_relation,
                templates={c: templates.get(c, "") for c in cs},
                decoding=dict(delta_f=dict(acc=acc_f, null_mean=nm_f, null_95=n95_f, p=p_f),
                              lexical=dict(acc=acc_e, null_mean=nm_e, null_95=n95_e, p=p_e)),
                identity_check_median=med)
    bundle = dict(cats=cs, axes=axes, t_global=t_global, write_centroids=centroids,
                  cos_peak=cos_peak, cos_early=cos_early, transfer=transfer, meta=meta)

    if a.all_layers:                          # depth timelapse source: cosine matrix per hidden layer
        nH = H_resid.shape[1]                 # L+1 hidden states (index 0 = embedding, b+1 = block b out)
        print(f"[all-layers] cosine matrix at every one of the {nH} hidden layers "
              "(cheap: reuses the cached states, no extra forward passes)...")
        cos_by_layer = torch.stack(
            [axis_cosine_matrix(H_resid[:, h, :].float(), cat_pairs)[1] for h in range(nH)], 0)
        bundle["cos_by_layer"] = cos_by_layer                 # [nH, K, K]
        bundle["meta"]["all_layers"] = True
        bundle["meta"]["n_hidden"] = nH
        bundle["meta"]["peak_hidden"] = peak + 1
        print(f"             cos_by_layer shape = {tuple(cos_by_layer.shape)} "
              "(saved in the .pt; visualize with visualizza_dizionario.py --timelapse)")

    if a.flip_layers:                         # the FFN pro->anti-truth flip at peak+1
        print("[flip-layers] FFN/attn contribution gap on the fixed peak axis, per block...")
        bundle["flip"] = flip_curves(H_resid, H_attn, H_ffn, pidx, peak, a.folds, a.seed)
        dpf = bundle["flip"]["dprime_ffn"]; fb = peak + 1
        if fb < len(dpf):
            print(f"             d'_ffn @peak {peak} = {dpf[peak]:+.2f} (pro-truth)   "
                  f"@block {fb} = {dpf[fb]:+.2f} ({'ANTI-truth flip' if dpf[fb] < 0 else 'still pro'})")

    os.makedirs(a.out_dir, exist_ok=True)     # existence was gated at the top (fail-fast)
    torch.save(bundle, out)
    with open(out.replace(".pt", ".json"), "w", encoding="utf-8") as f:
        json.dump(dict(meta, cats=cs,
                       cos_peak=[[round(float(x), 3) for x in r] for r in cos_peak],
                       cos_early=[[round(float(x), 3) for x in r] for r in cos_early],
                       transfer=[[round(float(x), 3) for x in r] for r in transfer]),
                  f, indent=2)
    print(f"\n[saved] {out}  (+ .json human summary)")
    print(f"  cats: {cs}")
    print(f"  axes [K,d] = {tuple(axes.shape)}   t_global [d] = {tuple(t_global.shape)}")
    print(f"  decoding: Delta f {acc_f:.2%} vs lexical {acc_e:.2%} (chance {1/len(cs):.0%})")
    print("  load with:  b = torch.load(path); scores = states @ b['axes'].T")
    return bundle


# =====================================================================
#  CLI
# =====================================================================
def resolve_jobs(a):
    """(model_name, peak, write_layer, early_block) for each requested model."""
    if a.models:
        jobs = []
        for name in a.models:
            cfg = MODELS.get(name)
            if cfg is not None:
                jobs.append((name,
                             a.peak if a.peak is not None else cfg["peak"],
                             a.write_layer if a.write_layer is not None else cfg["write_layer"],
                             a.early_block if a.early_block is not None else cfg["early_block"]))
            elif a.peak is not None and a.write_layer is not None:
                jobs.append((name, a.peak, a.write_layer,
                             a.early_block if a.early_block is not None else EARLY_BLOCK))
            else:
                raise SystemExit(
                    f"[error] '{name}' is not in the MODELS registry. Add it to the USER "
                    "CONFIG block, or pass --peak and --write-layer (measure the peak with a "
                    "per-layer truth-axis signal scan first -- it is model-specific).")
        return jobs
    return [(name, cfg["peak"], cfg["write_layer"], cfg["early_block"])
            for name, cfg in MODELS.items()]


def list_relations(a):
    """List every CounterFact relation (id, human template, unique-pair count) to
    a clean text file, so you can CHOOSE which to use as categories. No model is
    loaded -- this only reads the dataset. Paste the relation_ids you want into
    RELATION_WHITELIST (or set K_RELATIONS) in the USER CONFIG block above."""
    if a.file_counterfact:
        ds = open_local(a.file_counterfact)
    else:
        from datasets import load_dataset
        ds = load_dataset(DATASET_REPO, split="train", revision=a.rev_counterfact)
    print(f"[list-relations] {DATASET_REPO} @ {a.rev_counterfact} ({len(ds)} rows)")
    by_rel, templ = {}, {}
    for ex in ds:
        rid = str(ex["relation_id"])
        prompt = str(ex["prompt"]).strip()
        tt, tf = str(ex["target_true"]), str(ex["target_false"])
        if not prompt or tt.strip() == tf.strip():
            continue
        by_rel.setdefault(rid, set()).add((prompt, tt.strip()))   # unique-pair key (as in the builder)
        templ.setdefault(rid, str(ex["relation"]))
    ranked = sorted(by_rel, key=lambda r: len(by_rel[r]), reverse=True)

    L = []
    L.append("=" * 74)
    L.append("CounterFact relations  --  candidate categories for the truth dictionary")
    L.append("=" * 74)
    L.append(f"dataset  : {DATASET_REPO}")
    L.append(f"revision : {a.rev_counterfact}")
    L.append(f"relations: {len(ranked)}   (sorted by unique-pair count, descending)")
    L.append(f"legend   : '*' = current top-{a.k_relations} default    "
             f"ok = has >= PAIRS_PER_RELATION ({a.pairs_per_relation}) pairs")
    L.append("")
    L.append(f"  {'#':>3}  {'relation_id':<10} {'pairs':>6}  {'ok':<3}  template")
    L.append(f"  {'-'*3}  {'-'*10} {'-'*6}  {'-'*3}  {'-'*42}")
    for i, rid in enumerate(ranked, 1):
        star = "* " if i <= a.k_relations else "  "
        ok = "OK" if len(by_rel[rid]) >= a.pairs_per_relation else "low"
        L.append(f"{star}{i:>3}  {rid:<10} {len(by_rel[rid]):>6}  {ok:<3}  {templ[rid]}")
    L.append("")
    L.append("HOW TO USE (USER CONFIG block at the top of crea_dizionario.py):")
    L.append(f"  * keep RELATION_WHITELIST = None  -> auto top-{a.k_relations} (rows marked '*').")
    L.append("  * to FORCE an exact set, paste the ids you want, e.g.:")
    L.append(f"      RELATION_WHITELIST = {ranked[:a.k_relations]!r}")
    L.append("  * change how many categories with K_RELATIONS, pairs each with PAIRS_PER_RELATION.")
    text = "\n".join(L) + "\n"

    print("\n" + text)
    with open(a.list_out, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"[saved] {a.list_out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="*", default=None,
                    help="HF model names to export (default: every model in the registry)")
    ap.add_argument("--peak", type=int, default=None, help="override peak block (new/single model)")
    ap.add_argument("--write-layer", type=int, default=None, help="override FFN write layer")
    ap.add_argument("--early-block", type=int, default=None, help="override early/surface block")
    ap.add_argument("--k-relations", type=int, default=K_RELATIONS)
    ap.add_argument("--pairs-per-relation", type=int, default=PAIRS_PER_RELATION)
    ap.add_argument("--all-layers", action="store_true",
                    help="also compute the cosine matrix at EVERY hidden layer (saved as "
                         "'cos_by_layer' in the .pt) -- the source for the depth timelapse")
    ap.add_argument("--flip-layers", action="store_true",
                    help="also compute the FFN/attn contribution gap on the fixed peak axis "
                         "per block (saved as 'flip' in the .pt) -- the pro->anti-truth flip")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--perm", type=int, default=100)
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16", "float16"])
    ap.add_argument("--no-offload", action="store_true",
                    help="disable device_map CPU offload; on GPU OOM fall back to plain CPU")
    ap.add_argument("--rev-counterfact", default=DATASET_REVISION)
    ap.add_argument("--file-counterfact", default=None, help="local parquet/json/csv (offline)")
    ap.add_argument("--list-relations", action="store_true",
                    help="only LIST the dataset's relations (candidate categories) to a text "
                         "file, then exit -- no model is loaded")
    ap.add_argument("--list-out", default="relazioni_counterfact.txt",
                    help="output text file for --list-relations")
    ap.add_argument("--out-dir", default="dizionari",
                    help="output folder (kept separate so existing files are never overwritten)")
    ap.add_argument("--force", action="store_true", help="overwrite an existing output bundle")
    ap.add_argument("--verify", action="store_true",
                    help="compare the fresh cosine matrices to the embedded canonical K=8 archive")
    ap.add_argument("--verify-tol", type=float, default=0.02)
    a = ap.parse_args()

    if a.list_relations:                     # list categories to a file, then stop
        list_relations(a)
        return

    jobs = resolve_jobs(a)
    print(f"[plan] {len(jobs)} model(s): " + ", ".join(name for name, *_ in jobs))
    for name, peak, wl, eb in jobs:
        bundle = build_and_export(name, peak, wl, eb, a)
        if bundle is not None and a.verify:
            verify_against_reference(name, bundle, a.verify_tol)
    print(f"\n[done] dictionaries in ./{a.out_dir}/"
          + ("  (compare the MATCH/DIVERGES lines above)" if a.verify else ""))


if __name__ == "__main__":
    main()
