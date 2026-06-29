# How Much of Truth Fits on a Single Axis?

### Knowledge-Dependent Dimensionality and Unsupervised Polarity Recovery in Large Language Model Representations

This repository contains the implementation, paper, and follow-up working note
for an unsupervised geometric study of how factual truth is represented in a
large language model (Qwen2.5-1.5B).

---

## Core findings

1. **Unsupervised truth signal.** A geometric axis extracted via SVD of paired
   true/false differences carries a real truth signal, reaching a held-out AUC of
   up to 0.938 on clean minimal pairs (permutation p = 0.005).
2. **Knowledge-dependent dimensionality.** The gap between the unsupervised axis
   and a full linear probe scales with how well the model knows the fact:
   ~0.05 on known facts, ~0.20 on unknown ones.
3. **Unsupervised polarity recovery.** Unsupervised SVD on mixed-polarity data
   recovers the supervised polarity direction of Bürger et al. at cosine 0.959,
   without ever seeing polarity labels.

The one-dimensional reading was stress-tested with **seven independent
falsification attempts**, none of which overturned it.

---

## Repository structure

```
.
├── paper/
│   ├── truth_axis_arxiv.pdf
│   └── anatomy_note.pdf
├── src/
│   ├── truth_probe.py           # main reproduction script (positive results)
│   ├── anatomy.py               # residual-stream decomposition (attn vs FFN)
│   ├── anatomy_consolidate.py   # multi-seed stability of the decomposition
│   ├── anatomy_expanded.py      # truth axis inside the FFN expanded space
│   ├── ablation.py              # causal ablation of attention / FFN
│   └── canvas.py                # visualization helper
├── README.md
└── LICENSE
```

---

## Requirements

```
python >= 3.10
torch
transformers
datasets
```

Install:

```
pip install torch transformers datasets
```

Anatomy and ablation scripts load the model in float32 (the attention/FFN
decomposition identity requires the precision); ~6 GB of VRAM is enough for
Qwen2.5-1.5B.

---

## Reproducing the main results

All commands are run from `src/`. The CounterFact dataset revision is pinned for
exact reproducibility.

**Positive results (the paper's main claims):**

```
python truth_probe.py signal
python truth_probe.py polarity
python truth_probe.py recovery
```

**Anatomy — where the truth axis is born (attention vs FFN), per layer:**

```
python anatomy.py --dataset builtin
python anatomy.py --dataset counterfact --max-pairs 250 --perm 100 \
    --rev-counterfact c945b082ca08d0a8f3ba227fb78404a09614c36e
```

**Anatomy — is the attn/FFN picture stable across seeds:**

```
python anatomy_consolidate.py --dataset counterfact --max-pairs 250 --seeds 5 \
    --rev-counterfact c945b082ca08d0a8f3ba227fb78404a09614c36e
```

**Anatomy — truth axis inside the FFN expanded space (8960-dim):**

```
python anatomy_expanded.py --dataset counterfact --max-pairs 250 --perm 100 \
    --rev-counterfact c945b082ca08d0a8f3ba227fb78404a09614c36e
```

**Causal ablation — is the dominant component necessary:**

```
# middle band (attention builds the signal)
python ablation.py --dataset counterfact --max-pairs 250 \
    --band-start 11 --band-end 15 --readout 15 \
    --rev-counterfact c945b082ca08d0a8f3ba227fb78404a09614c36e

# transition band (the FFN degrades it)
python ablation.py --dataset counterfact --max-pairs 250 \
    --band-start 16 --band-end 18 --readout 18 \
    --rev-counterfact c945b082ca08d0a8f3ba227fb78404a09614c36e
```

Each anatomy/ablation tool embeds a self-check (decomposition identity,
permutation null, or across-seed stability) that fails loudly on bad data, so the
numbers it prints can be trusted only when that check passes.

---

## The follow-up note

`paper/anatomy_note.tex` documents an exploratory, mechanistic follow-up: how
attention and the FFN build, carry, and erode the truth signal across layers.
It is a **working note**, not a validated result to the standard of the main
paper, and is labeled as such.

---

## Citation

If you use this work, please cite the archived release:

> Vicidomini, Francesco Karim *How Much of Truth Fits on a Single Axis? Knowledge-Dependent
> Dimensionality and Unsupervised Polarity Recovery in LLM Representations.*
> Zenodo. DOI: 10.5281/zenodo.20938285

---

## License

- **Code** (`src/`): Apache-2.0 (see `LICENSE`).
- **Paper and note** (`paper/`): CC BY 4.0.
