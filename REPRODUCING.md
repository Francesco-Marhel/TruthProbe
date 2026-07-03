# Reproducing the results

All commands are run from `src/`. Dataset revisions are pinned for exact
reproducibility. Every tool embeds its own shields (identity checks,
held-out CV over pairs, permutation nulls, seed-stability criteria) and
prints its verdict.

> **Windows / PowerShell note:** the trailing `\` is bash line
> continuation. On PowerShell, join each multi-line command onto a single
> line (or replace `\` with a backtick `` ` ``).

Pinned revisions used everywhere below:

```
CounterFact : c945b082ca08d0a8f3ba227fb78404a09614c36e
TruthfulQA  : 741b8276f2d1982aa3d5b832d3ee81ed3b896490
```

---

## Paper 1 — How Much of Truth Fits on a Single Axis?

Positive results (Tables 1, 2):

```
python truth_probe.py signal --dataset builtin --with-2d --baseline --perm 200

python truth_probe.py signal --dataset mix --max-pairs 250 \
    --with-2d --baseline --perm 200 \
    --rev-counterfact c945b082ca08d0a8f3ba227fb78404a09614c36e \
    --rev-truthfulqa 741b8276f2d1982aa3d5b832d3ee81ed3b896490

python truth_probe.py polarity
python truth_probe.py recovery
```

Controlled falsification attempts (Table 3, attempts 2–4; attempt 1 is the
PHASE column of `signal` above):

```
python truth_probe.py domino --dataset builtin --perm 100

python truth_probe.py domino --dataset mix --max-pairs 250 --signal-mag --perm 100 \
    --rev-counterfact c945b082ca08d0a8f3ba227fb78404a09614c36e \
    --rev-truthfulqa 741b8276f2d1982aa3d5b832d3ee81ed3b896490

python truth_probe.py domino --dataset mix --max-pairs 250 --per-layer-axis --perm 100 \
    --rev-counterfact c945b082ca08d0a8f3ba227fb78404a09614c36e \
    --rev-truthfulqa 741b8276f2d1982aa3d5b832d3ee81ed3b896490
```

Visual observations (Figs. 1–3, attempts 5–7). Run the two datasets
separately; do not pass `--dataset mix` to `canvas.py` (mixing injects a
spurious sentence-format axis):

```
python canvas.py --dataset counterfact --layer 16 --out counterfact \
    --rev-counterfact c945b082ca08d0a8f3ba227fb78404a09614c36e

python canvas.py --dataset truthfulqa --layer 16 --out truthfulqa \
    --rev-truthfulqa 741b8276f2d1982aa3d5b832d3ee81ed3b896490
```

---

## Note 1 — Anatomy of the Truth Axis in Qwen2.5-1.5B

Anatomy and ablation scripts load the model in float32 (the attention/FFN
decomposition identity requires the precision); ~6 GB of VRAM is enough for
Qwen2.5-1.5B.

Where the truth axis is born (attention vs FFN), per layer:

```
python anatomy.py --dataset builtin
python anatomy.py --dataset counterfact --max-pairs 250 --perm 100 \
    --rev-counterfact c945b082ca08d0a8f3ba227fb78404a09614c36e
```

Is the attn/FFN picture stable across seeds:

```
python anatomy_consolidate.py --dataset counterfact --max-pairs 250 --seeds 5 \
    --rev-counterfact c945b082ca08d0a8f3ba227fb78404a09614c36e
```

Truth axis inside the FFN expanded space (8960-dim; trust the margin over
the elevated null, not the raw AUC):

```
python anatomy_expanded.py --dataset counterfact --max-pairs 250 --perm 100 \
    --rev-counterfact c945b082ca08d0a8f3ba227fb78404a09614c36e
```

Causal ablation, the three bands of the note:

```
# middle band (attention builds the signal)
python ablation.py --dataset counterfact --max-pairs 250 \
    --band-start 11 --band-end 15 --readout 15 \
    --rev-counterfact c945b082ca08d0a8f3ba227fb78404a09614c36e

# transition band (the FFN degrades the signal)
python ablation.py --dataset counterfact --max-pairs 250 \
    --band-start 16 --band-end 18 --readout 18 \
    --rev-counterfact c945b082ca08d0a8f3ba227fb78404a09614c36e

# late band (nothing remains to destroy)
python ablation.py --dataset counterfact --max-pairs 250 \
    --band-start 19 --band-end 26 --readout 27 \
    --rev-counterfact c945b082ca08d0a8f3ba227fb78404a09614c36e
```

---

## Note 2 — The FFN Flip is Peak-Relative

The CounterFact revision is the default in every note-2 script; the flags
below are shown explicitly only where they differ per model. Decomposition
runs are float32 (the 3B models spill to system RAM on a 12 GB GPU; the
in-code identity check certifies the run either way).

### Qwen2.5-1.5B (peak block 15, flip 16, band 16–18)

```
# who writes how much, and what, along the fixed truth axis
python ffn_erosion.py contrib

# corrected ablation: refit vs fixed-axis vs frozen control
python ffn_erosion.py ablate

# which layer's FFN erodes (single + cumulative)
python ffn_erosion.py sweep

# qualitative logit-lens of the band FFN contributions
python ffn_erosion.py lens

# exact gate/value split of the FFN's truth-axis gap (Eq. 1 of the note)
python swiglu.py attrib

# causal: freeze gate variation vs value variation in the band
python swiglu.py gatefreeze

# framing 2x2 (moral/prudential; lexical confound declared in the note)
python swiglu.py framing

# gauge-invariant static QK/OV circuit norms (weights only, CPU is fine)
python circuits.py
```

### Qwen2.5-3B (peak block 16, flip 17, band 17–19)

```
python truth_probe.py signal --dataset counterfact --max-pairs 250 \
    --model Qwen/Qwen2.5-3B --baseline --perm 100 \
    --rev-counterfact c945b082ca08d0a8f3ba227fb78404a09614c36e

python ffn_erosion.py contrib --model Qwen/Qwen2.5-3B \
    --axis-block 16 --band-start 17 --band-end 19 --scan-start 12 --scan-end 23

# 5 seeds + permutation null + rotation-vs-erosion check, one extraction
python flip_consolidate.py --model Qwen/Qwen2.5-3B

python ffn_erosion.py ablate --model Qwen/Qwen2.5-3B \
    --band-start 17 --band-end 19 --readout 19

python circuits.py --model Qwen/Qwen2.5-3B --dtype bfloat16
```

### Llama-3.2-1B (peak block 7, flip 8; gated repo, see README)

```
# behavior BEFORE geometry: the KNOWN rate pre-registers the expected gap
python behav_check.py --model meta-llama/Llama-3.2-1B

python truth_probe.py signal --dataset counterfact --max-pairs 250 \
    --model meta-llama/Llama-3.2-1B --baseline --with-2d --perm 100 \
    --rev-counterfact c945b082ca08d0a8f3ba227fb78404a09614c36e

python flip_consolidate.py --model meta-llama/Llama-3.2-1B \
    --axis-block 7 --flip-layer 8 --scan-start 3 --scan-end 13
```

### Figure 1 (peak-aligned collapse)

The collapse figure is built from the `d'_ffn` columns of the three
`contrib` / `flip_consolidate` tables above, aligned at each model's peak
block (x = layer − peak) and, in the right panel, normalized by the flip
amplitude |d'(peak+1)|.
