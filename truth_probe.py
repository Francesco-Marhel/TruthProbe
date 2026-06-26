# -*- coding: utf-8 -*-
"""
truth_probe.py  --  Single reproduction script for:
    "How Much of Truth Fits on a Single Axis?
     Knowledge-Dependent Dimensionality and Unsupervised Polarity Recovery in LLM Representations"

Every number in the paper is reproduced by ONE isolated subcommand. Running one task does
not load, extract, or compute anything for the others.

    python truth_probe.py signal    --dataset builtin                 # Table 1, clean pairs
    python truth_probe.py signal    --dataset mix --max-pairs 250     # Table 1, CounterFact/TruthfulQA
    python truth_probe.py polarity                                    # Table 2 (top): the polarity flip
    python truth_probe.py recovery                                    # Table 2 (bottom): unsupervised t_P

Add  -h  to any subcommand for its options (e.g.  python truth_probe.py signal -h).

Security: HuggingFace datasets are loaded as Parquet only, never with trust_remote_code;
the model is loaded with safetensors only. You may pin an exact commit with --rev-* or
load from a local file with --file-*. See the paper, Section "Reproducibility".

Method, in one paragraph: extract last-token hidden states at every layer in a single
forward pass. Build an UNSUPERVISED axis from the SVD of intra-pair (true - false)
differences (the shared topic cancels). Evaluate strictly out-of-fold (k-fold over pairs,
never splitting a pair), fit axis/calibration/layer-choice on train only, and quantify
selection bias with a label-permutation null. No dictionary, no trained probe for the axis.
"""
import os
import math
import argparse
import random
import torch


# =====================================================================
#  SHARED: model + extraction
# =====================================================================
def resolve_device_dtype(device, dtype):
    dev = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
    if dtype == "auto":
        dt = torch.bfloat16 if dev == "cuda" else torch.float32
    else:
        dt = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[dtype]
    return dev, dt


def load_model(name, dtype, device):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name, trust_remote_code=False)
    kw = dict(use_safetensors=True, trust_remote_code=False)
    try:                                  # transformers recenti: dtype=
        model = AutoModelForCausalLM.from_pretrained(name, dtype=dtype, **kw)
    except TypeError:                     # transformers vecchie: torch_dtype=
        model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=dtype, **kw)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    model.to(device)
    return tok, model


@torch.no_grad()
def forward_hidden(model, tok, text, device, pool):
    ids = tok(text, return_tensors="pt").to(device)
    hs = model(**ids, output_hidden_states=True).hidden_states
    if pool == "last":
        vec = torch.stack([h[0, -1, :] for h in hs], 0)
    else:
        vec = torch.stack([h[0].mean(0) for h in hs], 0)
    return vec.float().cpu()


def collect(model, tok, items, device, pool):
    H = []
    for i, (_, txt) in enumerate(items):
        H.append(forward_hidden(model, tok, txt, device, pool))
        if (i + 1) % 10 == 0 or i + 1 == len(items):
            print(f"\r[extract] {i+1}/{len(items)} sentences", end="", flush=True)
    print()
    return torch.stack(H, 0)


# =====================================================================
#  SHARED: geometry + metrics  (pure, testable without a model)
# =====================================================================
def auc_score(s, y):
    s = s.float(); y = y.long()
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    gt = (pos[:, None] > neg[None, :]).float().mean()
    eq = (pos[:, None] == neg[None, :]).float().mean()
    return float(gt + 0.5 * eq)


def unit(v):
    return v / v.norm().clamp_min(1e-8)


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


def kfold_facts(n_facts, k, seed=0):
    idx = list(range(n_facts)); random.Random(seed).shuffle(idx)
    folds = [idx[i::k] for i in range(k)]
    for i in range(k):
        test = set(folds[i]); train = [f for f in idx if f not in test]
        yield train, sorted(folds[i])


# =====================================================================
#  DATA: clean minimal pairs (Table 1, builtin) and polarity 2x2 (Table 2)
# =====================================================================
DEFAULT_PAIRS = [
    ("The Sun is a star.", "The Sun is a planet."),
    ("The Earth orbits the Sun.", "The Sun orbits the Earth."),
    ("The Moon orbits the Earth.", "The Earth orbits the Moon."),
    ("Water boils at one hundred degrees Celsius.", "Water boils at fifty degrees Celsius."),
    ("Water freezes at zero degrees Celsius.", "Water freezes at fifty degrees Celsius."),
    ("Water is made of hydrogen and oxygen.", "Water is made of iron and gold."),
    ("The heart pumps blood.", "The heart pumps air."),
    ("The lungs are used for breathing.", "The lungs are used for digestion."),
    ("Humans have two eyes.", "Humans have five eyes."),
    ("Spiders have eight legs.", "Spiders have four legs."),
    ("Bees make honey.", "Bees make bread."),
    ("Cows produce milk.", "Cows produce wine."),
    ("Fish live in water.", "Fish live in fire."),
    ("Birds have feathers.", "Birds have fur."),
    ("Dogs bark.", "Dogs meow."),
    ("Penguins live in cold regions.", "Penguins live in hot deserts."),
    ("Ocean water is salty.", "Ocean water is sweet."),
    ("Rome is the capital of Italy.", "Rome is the capital of Spain."),
    ("Paris is the capital of France.", "Paris is the capital of Germany."),
    ("Tokyo is the capital of Japan.", "Tokyo is the capital of China."),
    ("London is the capital of England.", "London is the capital of France."),
    ("The Nile river is in Africa.", "The Nile river is in Australia."),
    ("Mount Everest is the tallest mountain.", "Mount Everest is the shortest mountain."),
    ("The Pacific is the largest ocean.", "The Pacific is the smallest ocean."),
    ("The Sahara is a desert.", "The Sahara is an ocean."),
    ("Two plus two equals four.", "Two plus two equals five."),
    ("Three times three equals nine.", "Three times three equals six."),
    ("A triangle has three sides.", "A triangle has five sides."),
    ("A week has seven days.", "A week has ten days."),
    ("A year has twelve months.", "A year has twenty months."),
    ("Ice is frozen water.", "Ice is frozen oil."),
    ("Fire is hot.", "Fire is cold."),
    ("Light travels faster than sound.", "Sound travels faster than light."),
    ("Gold is a metal.", "Gold is a gas."),
    ("Diamonds are very hard.", "Diamonds are very soft."),
    ("Blood is red.", "Blood is green."),
    ("Grass is green.", "Grass is blue."),
    ("Snow is white.", "Snow is black."),
    ("The Earth is round.", "The Earth is flat."),
    ("Whales are mammals.", "Whales are fish."),
    ("Lions are carnivores.", "Lions are herbivores."),
    ("Mercury is the closest planet to the Sun.", "Mercury is the farthest planet from the Sun."),
    ("An adult human has thirty-two teeth.", "An adult human has ten teeth."),
    ("Sound cannot travel in a vacuum.", "Sound travels easily in a vacuum."),
    ("Plants produce oxygen.", "Plants produce gold."),
]

# 2x2 polarity design: (AT, AF, NT, NF). NT is TRUE, NF is FALSE.
QUADS = [
    ("The Sun is a star.", "The Sun is a planet.", "The Sun is not a planet.", "The Sun is not a star."),
    ("Whales are mammals.", "Whales are fish.", "Whales are not fish.", "Whales are not mammals."),
    ("Gold is a metal.", "Gold is a gas.", "Gold is not a gas.", "Gold is not a metal."),
    ("Spiders have eight legs.", "Spiders have six legs.", "Spiders do not have six legs.", "Spiders do not have eight legs."),
    ("Fire is hot.", "Fire is cold.", "Fire is not cold.", "Fire is not hot."),
    ("Dogs bark.", "Dogs meow.", "Dogs do not meow.", "Dogs do not bark."),
    ("Ice is frozen water.", "Ice is frozen oil.", "Ice is not frozen oil.", "Ice is not frozen water."),
    ("Paris is in France.", "Paris is in Germany.", "Paris is not in Germany.", "Paris is not in France."),
    ("The Earth is round.", "The Earth is flat.", "The Earth is not flat.", "The Earth is not round."),
    ("Blood is red.", "Blood is green.", "Blood is not green.", "Blood is not red."),
    ("Penguins are birds.", "Penguins are reptiles.", "Penguins are not reptiles.", "Penguins are not birds."),
    ("Lions are carnivores.", "Lions are herbivores.", "Lions are not herbivores.", "Lions are not carnivores."),
    ("Snow is white.", "Snow is black.", "Snow is not black.", "Snow is not white."),
    ("Sugar is sweet.", "Sugar is sour.", "Sugar is not sour.", "Sugar is not sweet."),
    ("Bees make honey.", "Bees make milk.", "Bees do not make milk.", "Bees do not make honey."),
    ("A triangle has three sides.", "A triangle has four sides.", "A triangle does not have four sides.", "A triangle does not have three sides."),
    ("The Moon orbits the Earth.", "The Moon orbits Mars.", "The Moon does not orbit Mars.", "The Moon does not orbit the Earth."),
    ("Iron is a metal.", "Iron is a liquid.", "Iron is not a liquid.", "Iron is not a metal."),
    ("Cows eat grass.", "Cows eat meat.", "Cows do not eat meat.", "Cows do not eat grass."),
    ("Sharks live in the ocean.", "Sharks live in trees.", "Sharks do not live in trees.", "Sharks do not live in the ocean."),
    ("Rome is the capital of Italy.", "Rome is the capital of Spain.", "Rome is not the capital of Spain.", "Rome is not the capital of Italy."),
    ("A diamond is hard.", "A diamond is soft.", "A diamond is not soft.", "A diamond is not hard."),
    ("Salt is salty.", "Salt is sweet.", "Salt is not sweet.", "Salt is not salty."),
    ("A frog is an amphibian.", "A frog is a mammal.", "A frog is not a mammal.", "A frog is not an amphibian."),
    ("The daytime sky is blue.", "The daytime sky is purple.", "The daytime sky is not purple.", "The daytime sky is not blue."),
    ("Humans breathe oxygen.", "Humans breathe helium.", "Humans do not breathe helium.", "Humans do not breathe oxygen."),
    ("A tomato is a fruit.", "A tomato is a mineral.", "A tomato is not a mineral.", "A tomato is not a fruit."),
    ("A volcano erupts lava.", "A volcano erupts ice.", "A volcano does not erupt ice.", "A volcano does not erupt lava."),
    ("A bat is a mammal.", "A bat is a bird.", "A bat is not a bird.", "A bat is not a mammal."),
    ("Glass is transparent.", "Glass is opaque.", "Glass is not opaque.", "Glass is not transparent."),
    ("Mercury is a planet.", "Mercury is a star.", "Mercury is not a star.", "Mercury is not a planet."),
    ("Wood comes from trees.", "Wood comes from rocks.", "Wood does not come from rocks.", "Wood does not come from trees."),
    ("The heart pumps blood.", "The heart pumps air.", "The heart does not pump air.", "The heart does not pump blood."),
    ("Honey is sweet.", "Honey is bitter.", "Honey is not bitter.", "Honey is not sweet."),
    ("A circle is round.", "A circle is square.", "A circle is not square.", "A circle is not round."),
    ("Plants need sunlight.", "Plants need darkness.", "Plants do not need darkness.", "Plants do not need sunlight."),
]


def pairs_to_items(pairs):
    items, pidx = [], []
    for ts, fs in pairs:
        it = len(items); items.append((1, ts))
        iff = len(items); items.append((0, fs))
        pidx.append((it, iff))
    return items, pidx


def build_polarity_items():
    items, fact, is_true, is_neg = [], [], [], []
    for fi, (at, af, nt, nf) in enumerate(QUADS):
        for text, tr, ng in [(at, 1, 0), (af, 0, 0), (nt, 1, 1), (nf, 0, 1)]:
            items.append((tr, text)); fact.append(fi); is_true.append(tr); is_neg.append(ng)
    return items, torch.tensor(fact), torch.tensor(is_true), torch.tensor(is_neg)


# ---- HuggingFace datasets (Parquet-only, security-hardened) ----
def _open_local(local_file):
    from datasets import load_dataset
    ext = os.path.splitext(local_file)[1].lower().lstrip(".")
    fmt = {"parquet": "parquet", "json": "json", "jsonl": "json", "csv": "csv"}.get(ext)
    if fmt is None:
        raise ValueError(f"unsupported local extension: {ext}")
    return load_dataset(fmt, data_files=local_file, split="train")


def load_counterfact_pairs(max_pairs, seed=0, revision=None, local_file=None):
    repo = "NeelNanda/counterfact-tracing"
    if local_file:
        ds = _open_local(local_file)
    else:
        from datasets import load_dataset
        ds = load_dataset(repo, split="train", revision=revision)   # no trust_remote_code
    print(f"  [provenance] {repo} @ {revision or 'latest'}  ({len(ds)} rows)")
    idx = list(range(len(ds))); random.Random(seed).shuffle(idx)
    pairs, seen = [], set()
    for i in idx:
        ex = ds[i]
        prompt = str(ex["prompt"]).strip()
        tt = str(ex["target_true"]).strip(); tf = str(ex["target_false"]).strip()
        if not prompt or not tt or not tf or tt == tf:
            continue
        key = (prompt, tt, tf)
        if key in seen:
            continue
        seen.add(key)
        pairs.append((f"{prompt} {tt}.", f"{prompt} {tf}."))
        if len(pairs) >= max_pairs:
            break
    if not pairs:
        raise RuntimeError("counterfact: no pairs built (schema changed?).")
    return pairs


def load_truthfulqa_pairs(max_pairs, seed=0, revision=None, local_file=None):
    repo = "truthful_qa"
    if local_file:
        ds = _open_local(local_file)
    else:
        from datasets import load_dataset
        try:
            ds = load_dataset(repo, "generation", split="validation", revision=revision)
        except Exception:
            repo = "truthfulqa/truthful_qa"
            ds = load_dataset(repo, "generation", split="validation", revision=revision)
    print(f"  [provenance] {repo} @ {revision or 'latest'}  ({len(ds)} rows)")
    rng = random.Random(seed)
    idx = list(range(len(ds))); rng.shuffle(idx)
    pairs = []
    for i in idx:
        ex = ds[i]
        q = str(ex["question"]).strip()
        best = str(ex.get("best_answer") or "").strip()
        corr = [str(a).strip() for a in (ex.get("correct_answers") or []) if str(a).strip()]
        wrong = [str(a).strip() for a in (ex.get("incorrect_answers") or []) if str(a).strip()]
        good = best if best else (corr[0] if corr else "")
        if not q or not good or not wrong:
            continue
        bad = rng.choice(wrong)
        if good.lower() == bad.lower():
            continue
        pairs.append((f"Q: {q} A: {good}", f"Q: {q} A: {bad}"))
        if len(pairs) >= max_pairs:
            break
    if not pairs:
        raise RuntimeError("truthfulqa: no pairs built (schema changed?).")
    return pairs


def load_pairs(source, max_pairs, seed, rev_cf, rev_tqa, file_cf, file_tqa):
    if source == "builtin":
        return list(DEFAULT_PAIRS)
    if source == "counterfact":
        return load_counterfact_pairs(max_pairs, seed, rev_cf, file_cf)
    if source == "truthfulqa":
        return load_truthfulqa_pairs(max_pairs, seed, rev_tqa, file_tqa)
    if source == "mix":
        h = max(1, max_pairs // 2)
        return (load_counterfact_pairs(h, seed, rev_cf, file_cf)
                + load_truthfulqa_pairs(max_pairs - h, seed + 1, rev_tqa, file_tqa))
    raise ValueError(source)


# =====================================================================
#  TASK 1: signal  -- Table 1 (held-out AUC, ablation, baseline, permutation)
# =====================================================================
def fit_axis(Hl, pidx, orient=None):
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


def fit_logit2d(Re, Im, y, iters=300, lr=0.5, l2=1e-3):
    X = torch.stack([Re, Im], 1).float(); yf = y.float()
    w = torch.zeros(2, requires_grad=True); b = torch.zeros(1, requires_grad=True)
    opt = torch.optim.Adam([w, b], lr=lr)
    for _ in range(iters):
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(X @ w + b, yf) + l2 * (w ** 2).sum()
        loss.backward(); opt.step()
    return w.detach(), float(b.detach())


def scores_all(Hl, ax, logit2d=None):
    f = project_fields(Hl, ax)
    out = {"1D": f["Re"], "MAG": -f["risk"], "PHASE": -f["phase_dev"]}
    if logit2d is not None:
        w, bsc = logit2d
        out["2D"] = w[0] * f["Re"] + w[1] * f["Im"] + bsc
    return out


def evaluate_layer_cv(Hl, pidx, k=5, seed=0, orient=None, with_2d=False):
    if orient is None:
        orient = [1] * len(pidx)
    keys = ["1D", "MAG", "PHASE"] + (["2D"] if with_2d else [])
    fold_auc = {kk: [] for kk in keys}
    for train_p, test_p in kfold_pairs(len(pidx), k, seed):
        ax = fit_axis(Hl, [pidx[p] for p in train_p], [orient[p] for p in train_p])
        logit2d = None
        if with_2d:
            si, sy = [], []
            for p in train_p:
                it, iff = pidx[p]
                if orient[p] >= 0: si += [it, iff]; sy += [1, 0]
                else: si += [iff, it]; sy += [1, 0]
            f = project_fields(Hl[si], ax)
            logit2d = fit_logit2d(f["Re"], f["Im"], torch.tensor(sy))
        ti, ty = [], []
        for p in test_p:
            it, iff = pidx[p]
            if orient[p] >= 0: ti += [it, iff]; ty += [1, 0]
            else: ti += [iff, it]; ty += [1, 0]
        sc = scores_all(Hl[ti], ax, logit2d)
        yt = torch.tensor(ty)
        for kk in keys:
            a = auc_score(sc[kk], yt)
            if a == a: fold_auc[kk].append(a)
    return {kk: (sum(v) / len(v) if v else float("nan")) for kk, v in fold_auc.items()}


def cv_curve(H, pidx, layers, k=5, seed=0, orient=None, with_2d=False):
    return {L: evaluate_layer_cv(H[:, L, :], pidx, k, seed, orient, with_2d) for L in layers}


def best_layer(curve, variant="1D"):
    bL, ba = None, -1.0
    for L, d in curve.items():
        a = d.get(variant, float("nan"))
        if a == a and a > ba: bL, ba = L, a
    return bL, ba


def permutation_null(H, pidx, layers, k=5, B=100, seed=0):
    rng = random.Random(seed); out = []
    for b in range(B):
        orient = [1 if rng.random() < 0.5 else -1 for _ in pidx]
        c = cv_curve(H, pidx, layers, k, seed + 1 + b, orient, with_2d=False)
        out.append(best_layer(c, "1D")[1])
        print(f"\r[permutation] {b+1}/{B}", end="", flush=True)
    print()
    return out


def probe_baseline_cv(Hl, pidx, y, k=5, seed=0, iters=300, lr=0.05, l2=1e-2):
    N, d = Hl.shape
    oof = torch.full((N,), float("nan"))
    for train_p, test_p in kfold_pairs(len(pidx), k, seed):
        tri, try_ = [], []
        for p in train_p:
            it, iff = pidx[p]; tri += [it, iff]; try_ += [1, 0]
        Xtr = Hl[tri].float(); mu, sd = Xtr.mean(0), Xtr.std(0).clamp_min(1e-6)
        Xtr = (Xtr - mu) / sd; ytr = torch.tensor(try_).float()
        w = torch.zeros(d, requires_grad=True); bsc = torch.zeros(1, requires_grad=True)
        opt = torch.optim.Adam([w, bsc], lr=lr)
        for _ in range(iters):
            opt.zero_grad()
            loss = torch.nn.functional.binary_cross_entropy_with_logits(Xtr @ w + bsc, ytr) + l2 * (w ** 2).sum()
            loss.backward(); opt.step()
        tei = []
        for p in test_p:
            it, iff = pidx[p]; tei += [it, iff]
        with torch.no_grad():
            oof[tei] = (Hl[tei].float() - mu) / sd @ w.detach() + bsc.detach()
    m = ~torch.isnan(oof)
    return auc_score(oof[m], y[m])


def cmd_signal(a):
    dev, dt = resolve_device_dtype(a.device, a.dtype)
    print(f"[task signal] dataset={a.dataset}  (Table 1)")
    pairs = load_pairs(a.dataset, a.max_pairs, a.seed, a.rev_counterfact,
                       a.rev_truthfulqa, a.file_counterfact, a.file_truthfulqa)
    items, pidx = pairs_to_items(pairs)
    y = torch.tensor([l for l, _ in items])
    print(f"[data] {len(items)} sentences = {len(pidx)} pairs")
    print(f"[model] {a.model} on {dev} ({dt})")
    tok, model = load_model(a.model, dt, dev)
    H = collect(model, tok, items, dev, a.pool)
    layers = list(range(H.shape[1]))

    print("\nlayer |  1D   |  MAG  | PHASE" + ("  |  2D" if a.with_2d else ""))
    curve = cv_curve(H, pidx, layers, a.folds, a.seed, with_2d=a.with_2d)
    for L in layers:
        d = curve[L]
        print(f"  {L:3d} | {d['1D']:.3f} | {d['MAG']:.3f} | {d['PHASE']:.3f}"
              + (f" | {d['2D']:.3f}" if a.with_2d else ""))
    bL, b1 = best_layer(curve, "1D")
    print(f"\nbest layer (1D, held-out): {bL}   AUC {b1:.3f}")
    d = curve[bL]
    print(f"ablation @layer {bL}: 1D={d['1D']:.3f} MAG={d['MAG']:.3f} PHASE={d['PHASE']:.3f}"
          + (f" 2D={d['2D']:.3f}  (2D-1D={d['2D']-d['1D']:+.3f})" if a.with_2d else ""))
    if a.baseline:
        ab = probe_baseline_cv(H[:, bL, :], pidx, y, a.folds, a.seed)
        print(f"baseline full linear probe @layer {bL}: {ab:.3f}   (geometry 1D: {b1:.3f})")
    if a.perm > 0:
        null = permutation_null(H, pidx, layers, a.folds, a.perm, a.seed)
        nt = torch.tensor(null); p = (1 + int((nt >= b1).sum())) / (a.perm + 1)
        print(f"permutation null: mean {nt.mean():.3f}  95pct {nt.quantile(0.95):.3f}  "
              f"observed {b1:.3f}  p={p:.4f}  -> "
              + ("REAL SIGNAL" if p < 0.05 else "not beyond selection bias"))


# =====================================================================
#  TASK 2: polarity  -- Table 2 (top): the affirmative->negated flip
# =====================================================================
def _cells(fact, is_true, is_neg, facts_subset):
    fs = set(int(f) for f in facts_subset)
    sel = torch.tensor([i for i in range(len(fact)) if int(fact[i]) in fs])
    def cell(ng, tr):
        return sel[[(int(is_neg[i]) == ng and int(is_true[i]) == tr) for i in sel]]
    return dict(AT=cell(0, 1), AF=cell(0, 0), NT=cell(1, 1), NF=cell(1, 0), ALL=sel)


def massmean(Hl, it, iff):
    return Hl[it].mean(0) - Hl[iff].mean(0)


def cmd_polarity(a):
    dev, dt = resolve_device_dtype(a.device, a.dtype)
    print("[task polarity] 2x2 affirmative/negated design  (Table 2, top)")
    items, fact, is_true, is_neg = build_polarity_items()
    print(f"[data] {len(items)} sentences = {int(fact.max())+1} facts x 4 cells")
    print(f"[model] {a.model} on {dev} ({dt})")
    tok, model = load_model(a.model, dt, dev)
    H = collect(model, tok, items, dev, a.pool)
    layers = list(range(H.shape[1]))
    n_facts = int(fact.max()) + 1

    rows = {}
    for L in layers:
        Hl = H[:, L, :].float()
        acc = dict(cosAN=[], aa=[], an=[], gn=[])
        for tr_f, te_f in kfold_facts(n_facts, a.folds, a.seed):
            c = _cells(fact, is_true, is_neg, tr_f); ct = _cells(fact, is_true, is_neg, te_f)
            if min(len(ct["AT"]), len(ct["AF"]), len(ct["NT"]), len(ct["NF"])) == 0:
                continue
            tA = massmean(Hl, c["AT"], c["AF"]); tN = massmean(Hl, c["NT"], c["NF"])
            tG = massmean(Hl, torch.cat([c["AT"], c["NT"]]), torch.cat([c["AF"], c["NF"]]))
            acc["cosAN"].append(float(torch.dot(unit(tA), unit(tN))))
            ia = torch.cat([ct["AT"], ct["AF"]]); inn = torch.cat([ct["NT"], ct["NF"]])
            acc["aa"].append(auc_score(Hl[ia] @ unit(tA), is_true[ia].long()))
            acc["an"].append(auc_score(Hl[inn] @ unit(tA), is_true[inn].long()))
            acc["gn"].append(auc_score(Hl[inn] @ unit(tG), is_true[inn].long()))
        rows[L] = {kk: (sum(v) / len(v) if v else float("nan")) for kk, v in acc.items()}

    print("\nlayer | cos(tA,tN) | aff->aff | aff->NEG | tG->NEG")
    for L in layers:
        r = rows[L]
        print(f"  {L:3d} |   {r['cosAN']:+.3f}   |  {r['aa']:.3f}  |  {r['an']:.3f}  |  {r['gn']:.3f}")
    best = max(layers, key=lambda L: rows[L]["aa"] if rows[L]["aa"] == rows[L]["aa"] else -1)
    r = rows[best]
    print(f"\nverdict @layer {best} (chosen on affirmative; negated stays OOD):")
    print(f"  cos(t_A, t_N)           : {r['cosAN']:+.3f}")
    print(f"  affirmative axis -> aff : {r['aa']:.3f}")
    print(f"  affirmative axis -> NEG : {r['an']:.3f}   (<0.5 = polarity flip)")
    print(f"  general t_G      -> NEG : {r['gn']:.3f}   (recovery)")


# =====================================================================
#  TASK 3: recovery  -- Table 2 (bottom): unsupervised SVD recovers t_P
# =====================================================================
def make_mixed_pairs(fact, is_true, is_neg, facts_subset):
    fs = set(int(f) for f in facts_subset); by = {}
    for i in range(len(fact)):
        if int(fact[i]) in fs:
            by.setdefault(int(fact[i]), {})[(int(is_neg[i]), int(is_true[i]))] = i
    it, iff = [], []
    for f, c in by.items():
        if (0, 1) in c and (0, 0) in c: it.append(c[(0, 1)]); iff.append(c[(0, 0)])
        if (1, 1) in c and (1, 0) in c: it.append(c[(1, 1)]); iff.append(c[(1, 0)])
    return torch.tensor(it), torch.tensor(iff)


def supervised_dirs(Hl, fact, is_true, is_neg, subset):
    c = _cells(fact, is_true, is_neg, torch.tensor(list(subset)))
    tA = Hl[c["AT"]].mean(0) - Hl[c["AF"]].mean(0)
    tN = Hl[c["NT"]].mean(0) - Hl[c["NF"]].mean(0)
    tG = Hl[torch.cat([c["AT"], c["NT"]])].mean(0) - Hl[torch.cat([c["AF"], c["NF"]])].mean(0)
    tP = tA - tN
    tGu = unit(tG); tPu = unit(tP - torch.dot(tP, tGu) * tGu)
    return tGu, tPu


def cmd_recovery(a):
    dev, dt = resolve_device_dtype(a.device, a.dtype)
    print("[task recovery] unsupervised SVD vs supervised t_P  (Table 2, bottom)")
    items, fact, is_true, is_neg = build_polarity_items()
    print(f"[data] {len(items)} sentences = {int(fact.max())+1} facts x 4 cells")
    print(f"[model] {a.model} on {dev} ({dt})")
    tok, model = load_model(a.model, dt, dev)
    H = collect(model, tok, items, dev, a.pool)
    layers = list(range(H.shape[1]))
    n_facts = int(fact.max()) + 1

    res = {}
    for L in layers:
        Hl = H[:, L, :].float()
        cg1, cpk, pck = [], [], []
        for s in range(a.splits):
            fl = list(range(n_facts)); random.Random(a.seed + s).shuffle(fl)
            sub = fl[: int(n_facts * 0.8)]
            it, iff = make_mixed_pairs(fact, is_true, is_neg, sub)
            D = Hl[it] - Hl[iff]                       # UNSUPERVISED: no polarity label
            _, _, Vh = torch.linalg.svd(D, full_matrices=False)
            V = Vh[: a.topk]
            tG, tP = supervised_dirs(Hl, fact, is_true, is_neg, sub)
            cg1.append(abs(float(torch.dot(unit(V[0]), tG))))
            coss = [abs(float(torch.dot(unit(V[j]), tP))) for j in range(V.shape[0])]
            cpk.append(max(coss)); pck.append(int(torch.tensor(coss).argmax()) + 1)
        res[L] = dict(cg1=sum(cg1) / len(cg1), cpk=sum(cpk) / len(cpk),
                      pc=round(sum(pck) / len(pck), 1))

    print("\nlayer | cos(SVD1,tG) | best|cos(SVD_k,tP)| | tP in PC#")
    for L in layers:
        r = res[L]
        print(f"  {L:3d} |    {r['cg1']:.3f}     |        {r['cpk']:.3f}         |   ~{r['pc']}")
    best = max(layers, key=lambda L: res[L]["cpk"])
    r = res[best]
    print(f"\nverdict @layer {best} (max t_P recovery):")
    print(f"  best |cos(SVD_k, t_P)| : {r['cpk']:.3f}  in PC ~{r['pc']}")
    print(f"  cos(SVD_1, t_G)        : {r['cg1']:.3f}")
    print("  -> " + ("unsupervised SVD recovers Burger's polarity direction without labels."
                     if r["cpk"] > 0.6 else "recovery weak/absent on this run."))


# =====================================================================
#  TASK 4: domino  -- OBSERVATIONAL. Does the depth signature beat a single layer?
#  (No runtime, no gate, no intervention. Pure measurement.)
#  Fixed axis = paired axis of the best layer, RE-READ across all layers.
# =====================================================================
def depth_signature(H, ax, layers):
    """Per sentence, the trajectory Delta_L = b_L - 1/2 read on the FIXED axis `ax`
    at every layer. Returns S in R^{N x nlayers}."""
    cols = []
    for L in layers:
        cols.append(project_fields(H[:, L, :], ax)["b"] - 0.5)
    return torch.stack(cols, 1)


def crossing_persistence(sig_row):
    """Descriptive only: first layer the trajectory goes below the line, and the
    fraction of subsequent layers it STAYS below (the 'domino' persistence)."""
    below = sig_row < 0
    if bool(below.any()):
        first = int(below.float().argmax())
        pers = float(below[first:].float().mean())
    else:
        first = len(sig_row); pers = 0.0
    return first, pers


def fit_logit_multi(X, y, iters=400, lr=0.1, l2=3e-2):
    """Logistic on a p-dim feature with strong L2 (the 29-dim signature can overfit
    on small data; heavy regularization keeps the comparison to 1 layer honest)."""
    p = X.shape[1]
    w = torch.zeros(p, requires_grad=True); b = torch.zeros(1, requires_grad=True)
    opt = torch.optim.Adam([w, b], lr=lr)
    yf = y.float()
    for _ in range(iters):
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(X @ w + b, yf) + l2 * (w ** 2).sum()
        loss.backward(); opt.step()
    return w.detach(), b.detach()


def best_train_layer(H, train_pairs, layers, orient_sub):
    bL, ba = layers[0], -1.0
    idx, yy = [], []
    for (it, iff), o in zip(train_pairs, orient_sub):
        if o >= 0: idx += [it, iff]; yy += [1, 0]
        else: idx += [iff, it]; yy += [1, 0]
    yt = torch.tensor(yy)
    for L in layers:
        ax = fit_axis(H[:, L, :], train_pairs, orient_sub)
        a = auc_score(project_fields(H[idx, L, :], ax)["Re"], yt)
        if a == a and a > ba: bL, ba = L, a
    return bL


def domino_fold_eval(H, pidx, layers, train_p, test_p, orient):
    """One fold: fit on train, return (local_auc, global_auc, per-test correctness +
    crossing/persistence by label)."""
    tp = [pidx[p] for p in train_p]; to = [orient[p] for p in train_p]
    bL = best_train_layer(H, tp, layers, to)
    ax = fit_axis(H[:, bL, :], tp, to)

    # signature on train (fit logistic) and standardize by train stats
    def idx_y(ps):
        I, Y = [], []
        for p in ps:
            it, iff = pidx[p]
            if orient[p] >= 0: I += [it, iff]; Y += [1, 0]
            else: I += [iff, it]; Y += [1, 0]
        return I, torch.tensor(Y)
    tri, trY = idx_y(train_p); tei, teY = idx_y(test_p)

    S_tr = depth_signature(H[tri], ax, layers)
    S_te = depth_signature(H[tei], ax, layers)
    mu, sd = S_tr.mean(0), S_tr.std(0).clamp_min(1e-6)
    w, b = fit_logit_multi((S_tr - mu) / sd, trY)
    g_te = ((S_te - mu) / sd) @ w + b                       # global score
    l_te = project_fields(H[tei, bL, :], ax)["Re"]          # local score (best layer)

    g_auc = auc_score(g_te, teY); l_auc = auc_score(l_te, teY)
    # per-sentence correctness (threshold at median of train scores)
    g_thr = (((S_tr - mu) / sd) @ w + b).median()
    l_thr = project_fields(H[tri, bL, :], ax)["Re"].median()
    g_corr = ((g_te > g_thr).long() == teY).float()
    l_corr = ((l_te > l_thr).long() == teY).float()
    # crossing / persistence (descriptive), by label, on the fixed axis
    rows = []
    for j in range(len(tei)):
        fr, pe = crossing_persistence(S_te[j])
        rows.append((int(teY[j]), fr, pe))
    return l_auc, g_auc, l_corr, g_corr, rows


def cmd_domino(a):
    dev, dt = resolve_device_dtype(a.device, a.dtype)
    print("[task domino] OBSERVATIONAL: does the depth signature beat a single layer?")
    print("              (no runtime, no gate, no intervention -- measurement only)")
    pairs = list(DEFAULT_PAIRS) if a.dataset == "builtin" else \
        load_pairs(a.dataset, a.max_pairs, a.seed, a.rev_counterfact,
                   a.rev_truthfulqa, a.file_counterfact, a.file_truthfulqa)
    items, pidx = pairs_to_items(pairs)
    print(f"[data] {len(items)} sentences = {len(pidx)} pairs")
    print(f"[model] {a.model} on {dev} ({dt})")
    tok, model = load_model(a.model, dt, dev)
    H = collect(model, tok, items, dev, a.pool)
    layers = list(range(H.shape[1]))
    orient = [1] * len(pidx)

    # --- observed: local vs global, held-out ---
    l_aucs, g_aucs, l_all, g_all, all_rows = [], [], [], [], []
    for tr_p, te_p in kfold_pairs(len(pidx), a.folds, a.seed):
        la, ga, lc, gc, rows = domino_fold_eval(H, pidx, layers, tr_p, te_p, orient)
        l_aucs.append(la); g_aucs.append(ga); l_all.append(lc); g_all.append(gc); all_rows += rows
    local = sum(l_aucs) / len(l_aucs); glob = sum(g_aucs) / len(g_aucs)
    diff = glob - local

    # error correlation between local and global (are they the same detector?)
    lc = torch.cat(l_all); gc = torch.cat(g_all)
    le, ge = 1 - lc, 1 - gc                                  # error indicators
    if le.std() > 0 and ge.std() > 0:
        err_corr = float(((le - le.mean()) * (ge - ge.mean())).mean() / (le.std() * ge.std()))
    else:
        err_corr = float("nan")

    # crossing / persistence by label (descriptive indicator, NOT causal)
    def by_label(lbl):
        sub = [(fr, pe) for (y, fr, pe) in all_rows if y == lbl]
        fr = sum(x[0] for x in sub) / len(sub); pe = sum(x[1] for x in sub) / len(sub)
        return fr, pe
    fr_t, pe_t = by_label(1); fr_f, pe_f = by_label(0)

    print("\n=== local (single best layer) vs global (depth signature), held-out ===")
    print(f"  local  AUC (single best layer) : {local:.3f}")
    print(f"  global AUC (depth signature)   : {glob:.3f}   ({diff:+.3f} vs local)")
    print(f"  error correlation local/global : {err_corr:+.3f}  "
          "(low = complementary; high = same detector)")
    print("\n=== crossing / persistence (DESCRIPTIVE indicator, not causal) ===")
    print(f"  true  sentences: first-cross layer ~{fr_t:.1f}, persistence below line {pe_t:.2f}")
    print(f"  false sentences: first-cross layer ~{fr_f:.1f}, persistence below line {pe_f:.2f}")
    print(f"  -> persistence gap (false - true): {pe_f - pe_t:+.2f}  "
          "(positive = false stays below once it crosses = domino)")

    # --- permutation null for the GLOBAL signature (overfitting / selection check) ---
    if a.perm > 0:
        print()
        rng = random.Random(a.seed); null = []
        for bptr in range(a.perm):
            o = [1 if rng.random() < 0.5 else -1 for _ in pidx]
            gg = []
            for tr_p, te_p in kfold_pairs(len(pidx), a.folds, a.seed):
                _, ga, _, _, _ = domino_fold_eval(H, pidx, layers, tr_p, te_p, o)
                gg.append(ga)
            null.append(sum(gg) / len(gg))
            print(f"\r[permutation] {bptr+1}/{a.perm}", end="", flush=True)
        print()
        nt = torch.tensor(null); p = (1 + int((nt >= glob).sum())) / (a.perm + 1)
        print(f"  permutation null (global): mean {nt.mean():.3f}  95pct {nt.quantile(0.95):.3f}  "
              f"observed {glob:.3f}  p={p:.4f}")
        if p >= 0.05:
            print("  -> global NOT beyond chance: the 29-dim signature is fitting noise.")
        elif diff > 0.01:
            print("  -> global is real AND beats the single layer: depth signal exists.")
        else:
            print("  -> global is real but does NOT beat the single layer: best layer already had it.")


# =====================================================================
#  CLI: isolated subcommands
# =====================================================================
def add_common(p):
    p.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
    p.add_argument("--pool", default="last", choices=["last", "mean"])
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--dtype", default="auto", choices=["auto", "float32", "float16", "bfloat16"])
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="task", required=True)

    ps = sub.add_parser("signal", help="Table 1: held-out AUC, ablation, baseline, permutation")
    add_common(ps)
    ps.add_argument("--dataset", default="builtin",
                    choices=["builtin", "counterfact", "truthfulqa", "mix"])
    ps.add_argument("--max-pairs", type=int, default=250)
    ps.add_argument("--with-2d", action="store_true")
    ps.add_argument("--baseline", action="store_true")
    ps.add_argument("--perm", type=int, default=200)
    ps.add_argument("--rev-counterfact", default=None)
    ps.add_argument("--rev-truthfulqa", default=None)
    ps.add_argument("--file-counterfact", default=None)
    ps.add_argument("--file-truthfulqa", default=None)
    ps.set_defaults(func=cmd_signal)

    pp = sub.add_parser("polarity", help="Table 2 (top): affirmative->negated flip")
    add_common(pp)
    pp.set_defaults(func=cmd_polarity, folds=6)   # leave-facts-out: default 6

    pr = sub.add_parser("recovery", help="Table 2 (bottom): unsupervised t_P recovery")
    add_common(pr)
    pr.add_argument("--splits", type=int, default=8)
    pr.add_argument("--topk", type=int, default=6)
    pr.set_defaults(func=cmd_recovery)

    pd = sub.add_parser("domino", help="Observational: does the depth signature beat a single layer?")
    add_common(pd)
    pd.add_argument("--dataset", default="builtin",
                    choices=["builtin", "counterfact", "truthfulqa", "mix"])
    pd.add_argument("--max-pairs", type=int, default=250)
    pd.add_argument("--perm", type=int, default=100)
    pd.add_argument("--rev-counterfact", default=None)
    pd.add_argument("--rev-truthfulqa", default=None)
    pd.add_argument("--file-counterfact", default=None)
    pd.add_argument("--file-truthfulqa", default=None)
    pd.set_defaults(func=cmd_domino)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
