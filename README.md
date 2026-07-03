# TruthProbe

Unsupervised geometry of truth in small language models: a training-free
truth axis, its knowledge-dependent dimensionality, and a cross-family
anatomy of how the network builds and erodes it.

## Papers (in `paper/`)

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
3. **The FFN Flip is Peak-Relative** (`ffn_flip_note.pdf`) — chapter 2,
   closing the anatomy. On three models across two families (Qwen2.5-1.5B,
   Qwen2.5-3B, Llama-3.2-1B) the FFN writes pro-truth into the truth peak
   and flips to stable anti-truth at exactly peak+1, confirmed by
   pre-registered prediction with permutation nulls. An exact SwiGLU split
   attributes the erosion to the value stream, not the gate. Two competing
   hypotheses are falsified by pre-registered criteria and reported as
   results.

Superseded versions live in `drafts/`.

## Layout

```
paper/          current papers (PDF) + figures/
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
- **Papers, notes and figures** (`paper/`, `drafts/`): CC BY 4.0.

## Citation

If you use this work, please cite the papers (Zenodo DOI
10.5281/zenodo.20938285 for paper 1; the note DOIs are listed in `paper/`
once minted) and link this repository.
