# TruthProbe

Training-free geometry of truth in small language models: a single-axis
truth direction (label-free up to sign, given the paired design), its
knowledge-dependent dimensionality, a relational law of how attention and
the FFN build and overwrite it, and the semantically signed category
mixture it summarizes. Four base models, two families, pre-registered
predictions throughout.

## The manuscript (in `Paper/`)

**The Anatomy of a Truth Direction: Knowledge-Dependent Dimensionality, a
Relational Law, and a Convergent Category Geometry in Small Language
Models** (`monolith.pdf`) consolidates the whole series in two parts:

- **Part I** (one model): the training-free axis from the SVD of
  hidden-state minimal pairs; held-out AUC up to 0.938 against a
  full-probe ceiling of 0.989; the axis-vs-probe gap grows as the model's
  behavioral knowledge decreases and as material heterogeneity increases;
  seven falsification experiments; recovery of the supervised polarity
  direction of Buerger et al. at cosine 0.959 without polarity labels.
- **Part II** (four models, two families): the relational law (FFN writes
  oppose every truth frame that does not contain them; attention
  propagates frames it did not write), narrated together with the
  correction of our own earlier peak-anchored claim; causal attribution of
  the post-peak erosion to the SwiGLU value stream on all four models;
  two pre-registered discriminations (tug-of-war is a trait of scale,
  ignition of one family); a pre-registered cross-family negative for the
  static OV/QK crossover; and the category geometry, whose arrangement
  converges across families (Mantel p = 0.0009).

Earlier standalone documents (the single-axis paper, the anatomy note,
the flip note v1/v2, the category note) are superseded by the manuscript;
their versions live in `drafts/`.


## Layout

```
Paper/          current manuscript (PDF) + figures/
drafts/         superseded working-note versions
src/            all scripts, flat; they import each other; do not split
                into subfolders or the imports (and the commands in
                REPRODUCING.md) break
index.html      (GitHub Pages)
README.md
REPRODUCING.md  full reproduction commands, one command per reported number
LICENSE         Apache 2.0 (code)
```

## Setup

```
pip install torch transformers datasets
```

A CUDA GPU is optional; 12 GB is enough for every run (decomposition runs
use float32 and spill to system RAM on the 3B models). Models load with
safetensors only and `trust_remote_code=False`; external datasets load as
Parquet only, at pinned revisions. `meta-llama/Llama-3.2-*` are gated:
accept the license on Hugging Face and authenticate once with
`hf auth login` (read token).

## Reproducing 

Every number in the manuscript is reproduced by one isolated command; the
full list, organized per part and per model, is in
**[REPRODUCING.md](REPRODUCING.md)**. Every tool embeds its own safeguards
(additive identity checks, held-out cross-validation over pairs,
permutation nulls, multi-seed stability criteria) and prints its
measurements in full; no tool prints an automatic verdict, and the model
is passed explicitly to every command. A run is certified by its in-code
checks, not by its launch conditions.
(All scripts in this repository were generated using a Large Language Model (LLM) under my direct guidance and supervision).

The calculated truth directions and extracted semantic weights are organized in the `Data/` directory, divided by model architecture. 
Each folder contains both the raw PyTorch weights (`.pt`) for direct implementation,
and human-readable text mappings (`.json`) for quick inspection.

## License

- **Code** (`src/`): Apache License 2.0 (see `LICENSE`).
- **Papers, notes and figures** (`Paper/`, `drafts/`): CC BY 4.0.

## Citation

Please cite the manuscript via its Zenodo deposit
(https://doi.org/10.5281/zenodo.21348060) and
link this repository.
