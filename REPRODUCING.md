# Reproducing the results

All commands are run from `src/`. Dataset revisions are pinned for exact
reproducibility. Every tool embeds its own safeguards (identity checks,
held-out CV over pairs, permutation nulls, seed-stability criteria) and
prints its measurements in full; no tool prints an automatic verdict.

> **Windows / PowerShell note:** the trailing `\` is bash line
> continuation. On PowerShell, join each multi-line command onto a single
> line (or replace `\` with a backtick `` ` ``).

Pinned revisions used everywhere below:

```
CounterFact : c945b082ca08d0a8f3ba227fb78404a09614c36e
TruthfulQA  : 741b8276f2d1982aa3d5b832d3ee81ed3b896490
```

## Golden rules (learned inside this project)

1. **Pass `--model` explicitly to every command.** Tool defaults differ:
   `flip_consolidate.py` defaults to Qwen2.5-3B, most other tools to
   Qwen2.5-1.5B. One omitted flag once consolidated the wrong model (the
   accident is reported in the manuscript; it accidentally confirmed the
   relational law off-peak).
2. **Landmarks are per model.** Use this table everywhere a command asks
   for a peak, flip, band, or hidden level:

```
model          peak block   flip   erosion band   readout   hidden level (peak+1)
Qwen2.5-1.5B       15         16      16-18          18            16
Qwen2.5-3B         16         17      17-19          19            17
Llama-3.2-1B        7          8       8-10          10             8
Llama-3.2-3B        9         10      10-12          12            10
```

3. Save one output file per run; write predictions to a file **before**
   launching runs on untested models.

---

## Part I (Qwen2.5-1.5B)

Signal, baselines, permutation null (Tables 1-2); polarity; recovery:

```
python truth_probe.py signal --model Qwen/Qwen2.5-1.5B --dataset builtin \
    --with-2d --baseline --perm 200
python truth_probe.py signal --model Qwen/Qwen2.5-1.5B --dataset mix --max-pairs 250 \
    --with-2d --baseline --perm 200 \
    --rev-counterfact c945b082ca08d0a8f3ba227fb78404a09614c36e \
    --rev-truthfulqa 741b8276f2d1982aa3d5b832d3ee81ed3b896490
python truth_probe.py polarity --model Qwen/Qwen2.5-1.5B
python truth_probe.py recovery --model Qwen/Qwen2.5-1.5B
```

Falsification experiments (domino variants; canvas for the visual ones,
one dataset at a time, never `--dataset mix` on canvas):

```
python truth_probe.py domino --model Qwen/Qwen2.5-1.5B --dataset builtin --perm 100
python truth_probe.py domino --model Qwen/Qwen2.5-1.5B --dataset mix --max-pairs 250 \
    --signal-mag --perm 100 \
    --rev-counterfact c945b082ca08d0a8f3ba227fb78404a09614c36e \
    --rev-truthfulqa 741b8276f2d1982aa3d5b832d3ee81ed3b896490
python truth_probe.py domino --model Qwen/Qwen2.5-1.5B --dataset mix --max-pairs 250 \
    --per-layer-axis --perm 100 \
    --rev-counterfact c945b082ca08d0a8f3ba227fb78404a09614c36e \
    --rev-truthfulqa 741b8276f2d1982aa3d5b832d3ee81ed3b896490
python canvas.py --dataset counterfact --layer 16 --out counterfact \
    --rev-counterfact c945b082ca08d0a8f3ba227fb78404a09614c36e
python canvas.py --dataset truthfulqa --layer 16 --out truthfulqa \
    --rev-truthfulqa 741b8276f2d1982aa3d5b832d3ee81ed3b896490
```

Anatomy (attention/FFN decomposition, ablations, expanded-space negative):

```
python anatomy.py --model Qwen/Qwen2.5-1.5B --dataset counterfact --max-pairs 250 \
    --perm 100 --rev-counterfact c945b082ca08d0a8f3ba227fb78404a09614c36e
python anatomy_consolidate.py --model Qwen/Qwen2.5-1.5B
python ablation.py --model Qwen/Qwen2.5-1.5B
python anatomy_expanded.py --model Qwen/Qwen2.5-1.5B
```

Norm-weighting control (Methods of Part I; repeat per model with its
hidden level from the table):

```
python axis_norm_check.py --model Qwen/Qwen2.5-1.5B --layer 16
```

---

## Part II (four models; repeat each block with each `--model` and its
landmarks from the table)

Phase 0: behavioral label BEFORE geometry, then peaks and the axis/probe
gap under the uniform protocol:

```
python behav_check.py --model meta-llama/Llama-3.2-3B --n 200 --seed 0
python truth_probe.py signal --model meta-llama/Llama-3.2-3B \
    --dataset counterfact --max-pairs 250 --baseline --perm 200 \
    --rev-counterfact c945b082ca08d0a8f3ba227fb78404a09614c36e
```

Phase 1: the flip, consolidated (5 seeds + permutation null + rotation
check; float32):

```
python flip_consolidate.py --model meta-llama/Llama-3.2-3B \
    --axis-block 9 --flip-layer 10 --scan-start 5 --scan-end 15
```

Phase 2: axis provenance, post and pre frames, both components:

```
python axis_provenance.py --model meta-llama/Llama-3.2-3B --peak 9 \
    --scan-start 5 --scan-end 15
python axis_provenance.py --model meta-llama/Llama-3.2-3B --peak 9 \
    --scan-start 5 --scan-end 15 --component attn
```

Phase 3: causal ablation over the erosion band; exact SwiGLU split and
causal freezes:

```
python ffn_erosion.py ablate --model meta-llama/Llama-3.2-3B \
    --band-start 10 --band-end 12 --readout 12
python swiglu.py attrib --model meta-llama/Llama-3.2-3B \
    --axis-block 9 --scan-start 6 --scan-end 13
python swiglu.py gatefreeze --model meta-llama/Llama-3.2-3B \
    --band-start 10 --band-end 12 --readout 12
```

Phase 4: gauge-invariant static circuit norms (weights only; CPU is
fine). Landmarks are annotations only; the norm tables do not depend on
them:

```
python circuits.py --model meta-llama/Llama-3.2-3B --dtype bfloat16 \
    --peak 9 --flip 10
```

Phase 5: category geometry (the registered falsification runs on the two
3B models; Llama-3.2-1B is excluded by the scoping rule stated in the
manuscript), arrangement law, dictionary export:

```
python categories.py --model Qwen/Qwen2.5-3B --peak 16 --write-layer 17 --k-relations 8
python categories.py --model meta-llama/Llama-3.2-3B --peak 9 --write-layer 10 --k-relations 8
python categories.py --model Qwen/Qwen2.5-1.5B --peak 15 --write-layer 16 --k-relations 5
python arrangement_law.py
python dictionary_export.py --model Qwen/Qwen2.5-3B --peak 16 --write-layer 17 --k-relations 8
python dictionary_export.py --model meta-llama/Llama-3.2-3B --peak 9 --write-layer 10 --k-relations 8
```

`arrangement_law.py` carries the canonical matrices transcribed in its
source, with the regeneration commands in its header; before trusting its
statistics after any change, diff the matrices printed by fresh
`categories.py` runs against the constants in the source, cell by cell.
Note the `--k-relations 8` on `dictionary_export.py`: the tool's default
is 5, and the manuscript's dictionaries are K = 8.

Synthetic validations of the intervention tooling (run anytime, no GPU):

```
python test_axis_provenance.py
python test_ffn_erosion.py
python test_swiglu.py
```
