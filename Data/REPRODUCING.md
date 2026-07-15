## 0. Setup (All scripts in this repository were generated using a Large Language Model (LLM) under my direct guidance and supervision).

- **Security**: models load with `use_safetensors=True` and `trust_remote_code=False`;
  the dataset (`NeelNanda/counterfact-tracing`) is read as **Parquet** at a **pinned
  revision** (`c945b082ca08d0a8f3ba227fb78404a09614c36e`) — reproducible, no remote code.
- **Precision**: the *axis / behavioural* reads run in **bf16** on GPU (the SVD axis is
  robust); the *decomposition / dictionary* scripts force **fp32** (the
  `attn + ffn == residual-delta` identity collapses in bf16). Keep these defaults.
- **GPU**: a 3B model in fp32 is ~12 GB. On a 12 GB card `crea_dizionario.py` auto-switches
  to `device_map='auto'` (GPU + CPU offload, fp32 preserved). Smaller models run fully on GPU.
- **Gated model**: `meta-llama/Llama-3.2-3B` requires being logged in to the HF Hub
  (`huggingface-cli login`), see `test_teoria/test_llama.py`.
- **Layout note**: the analysis scripts live in `test_teoria/`; they import each other, so
  run them by path (Python puts the script's folder on `sys.path`). The venv also carries
  `studio_paths.pth`, which makes `test_teoria/` importable from any working directory.

---

## 1. The truth dictionary — `crea_dizionario.py` (stand-alone)

Builds, per model, the measured dictionary bundle (`dizionari/truth_dictionary_<tag>.pt`
`+ .json`): per-category truth axes, global axis, FFN write centroids, signed cosine
matrices (peak + early surface control), held-out transfer matrix, and category-decoding
stats. Outputs go to `dizionari/` and never overwrite existing files (use `--force`).

> **Re-running a model you already built?** (e.g. to add `--all-layers` or
> `--flip-layers` to an existing bundle) The output is **not overwritten** — the run
> is skipped immediately with a `[skip] ... already exists` message. Add **`--force`**
> to overwrite it, otherwise the extra data is computed but never saved.

```
# choose categories: list all dataset relations (id, template, pair count) to a text file
python crea_dizionario.py --list-relations                     # -> relazioni_counterfact.txt
python crea_dizionario.py --list-relations --list-out my.txt

# build every model in the registry (both 3B), K=8, fp32
python crea_dizionario.py

# build one model
python crea_dizionario.py --models Qwen/Qwen2.5-3B

# build + check reproducibility vs the canonical K=8 archive (MATCH/DIVERGES)
python crea_dizionario.py --verify

# a NEW model (measure its peak first, then pass peak/write-layer)
python crea_dizionario.py --models <hf/org/name> --peak 20 --write-layer 21

# depth extras (both reuse the cached states, no extra forward passes):
#   --all-layers  adds 'cos_by_layer' (per-layer category cosine geometry -> timelapse)
#   --flip-layers adds 'flip' (FFN/attn gap on the fixed peak axis per block -> the b+1 flip)
python crea_dizionario.py --models Qwen/Qwen2.5-3B --all-layers --flip-layers

# useful flags
#   --k-relations N          number of categories (top-N relations)
#   --pairs-per-relation N   pairs sampled per relation (default 60)
#   --all-layers             also save per-layer cosine matrices (cos_by_layer) in the .pt
#   --flip-layers            also save the FFN pro->anti-truth flip per block ('flip') in the .pt
#   --dtype float32|bfloat16 keep float32 (bf16 aborts at the identity gate)
#   --no-offload             disable CPU offload; on GPU OOM fall back to plain CPU
#   --out-dir DIR            output folder (default: dizionari)
#   --force                  allow overwriting an existing bundle
```

**Expanding categories**: edit the `USER CONFIG` block at the top of the file —
`K_RELATIONS`, `PAIRS_PER_RELATION`, and `RELATION_WHITELIST` (paste specific relation
ids from `--list-relations` to pin an exact set).

---

## 2. Visualise the dictionary — `visualizza_dizionario.py` (stand-alone)

Reads a saved bundle (prefer the **.pt** — full precision, and the only format that
carries `cos_by_layer`) and produces a readable numeric summary + heatmaps, plus an
optional depth timelapse. Images go to `grafici/`. Colours are a diverging map with a
neutral midpoint (cosines centred at 0; transfer AUC centred at 0.5 = chance).

```
# numeric summary (off-diag mean/std, peak-vs-early, transfer within/cross, per-category
# rows) + heatmaps of cos_peak / cos_early / transfer
python visualizza_dizionario.py dizionari/truth_dictionary_Qwen25_3B.pt

# + depth timelapse: evolution curve (off-diag mean vs layer) and animated GIF
# (needs a .pt built with crea_dizionario.py --all-layers)
python visualizza_dizionario.py dizionari/truth_dictionary_Qwen25_3B.pt --timelapse

# + the FFN pro->anti-truth flip curve (needs a .pt built with --flip-layers)
python visualizza_dizionario.py dizionari/truth_dictionary_Qwen25_3B.pt --flip

# no path = newest .pt in dizionari/ ; other flags: --out-dir DIR --fps N --no-summary
python visualizza_dizionario.py --timelapse --flip --fps 4
```

> Two DIFFERENT depth views:
> - `--timelapse` animates the **per-category cosine geometry** (how the category
>   axes relate to one another, layer by layer) — from `--all-layers`.
> - `--flip` draws the **FFN pro→anti-truth flip**: the FFN contribution's class gap
>   on the fixed truth axis, pro-truth up to the peak and anti-truth from block b+1 —
>   from `--flip-layers`. Same quantity as `test_teoria/flip_consolidate.py`.

