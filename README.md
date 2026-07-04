# TruthProbe

Unsupervised geometry of truth in small language models: a training-free
truth axis, its knowledge-dependent dimensionality, and a cross-family
anatomy of how the network builds and erodes it.

Note for readers: If the built-in GitHub PDF viewer fails to load the previews or the download button hangs due to vector graphics complexity, you can download the full repository archive directly from Zenodo (Zenodo DOI
10.5281/zenodo.20938285).


## Papers (in `Paper/`)

1. **How Much of Truth Fits on a Single Axis?** (`truth_axis_arxiv.pdf`) —
   an unsupervised SVD truth axis at cost O(d); held-out AUC up to 0.938
   with permutation control; the axis-vs-probe gap scales with the model's
   knowledge of the fact; unsupervised recovery of the polarity direction of
   Bürger et al. at cosine 0.959; seven falsification attempts, none of
   which overturns the one-dimensional reading.
2. **Anatomy of the Truth Axis in Qwen2.5-1.5B** (`anatomy_note.pdf`) —
   working note, chapter 1 of the anatomy: attention/FFN decomposition of
   the residual stream, per-layer readability, the expanded-space negative,
   and the first causal ablations.
  3. **Attention Propagates, the FFN Overwrites** (`FFN_flip_v2.pdf`, v2) —
   chapter 2, closing the anatomy. An axis-provenance control generalized
   v1's peak-anchored flip into a relational law: the FFN's write at block b
   opposes every truth frame that does not contain it, while attention
   propagates existing frames — near mirror images at the peak (+1.63 vs
   −1.43 on Llama-3.2-3B). Confirmed by pre-registered prediction on four
   models across two families (Qwen2.5-1.5B/3B, Llama-3.2-1B/3B). The
   frame-free erosion, its causal value-stream attribution, and every v1
   measurement stand; the dimensionality law gains a fourth point. v1's
   peak-anchored interpretation is superseded (v1 in `drafts/`).

Superseded versions live in `drafts/`.

## Layout

```
Paper/          current papers (PDF) + figures/
drafts/         superseded working-note versions
src/            all scripts, flat — they import each other; do not split
                into subfolders or the imports (and the commands in
                REPRODUCING.md) break
README.md
REPRODUCING.md  full reproduction commands for all three documents
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

Every number in the three documents is reproduced by one isolated command;
the full list, organized per paper and per model, is in
**[REPRODUCING.md](REPRODUCING.md)**. Every tool embeds its own shields —
additive identity checks, held-out cross-validation over pairs, permutation
nulls, multi-seed stability criteria — and prints its verdict: a run is
certified by its in-code checks, not by its launch conditions.

## License

- **Code** (`src/`): Apache License 2.0 (see `LICENSE`).
- **Papers, notes and figures** (`Paper/`, `drafts/`): CC BY 4.0.

## Citation

If you use this work, please cite the papers (Zenodo DOI
10.5281/zenodo.20938285 for paper 1; the note DOIs are listed in `Paper/`
once minted) and link this repository.
