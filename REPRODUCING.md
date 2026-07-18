# Reproducing every number

## Golden rules (learned inside this project)

1. **The model is passed explicitly to every command.** Tool defaults
   differ; one default once produced an accidental replication.
2. **No tool prints an automatic verdict.** Tools print matrices,
   margins, and distances; the reading belongs to the researcher.
3. **Predictions are written to file before runs.** Failed predictions
   are reported with the same care as confirmed ones
   (`Data/predictions/`).
4. **Artifacts prove their identity.** Model, K, pairs, seed, and
   dataset revision live inside every exported file and in its
   filename; content hashes and cosine fingerprints settle disputes
   (`src/inventario_dizionari.py`, `src/identikit_pt.py`).
5. **Source folders are append-only.** Analyses run on a derived,
   deduplicated archive; unprovenanced files are quarantined, not
   repaired.
6. Decomposition runs use `float32` (the additive identity check fails
   spuriously in `bfloat16`); extraction-only runs use `bfloat16`.
7. Datasets load as Parquet at pinned revisions; models load with
   safetensors only, `trust_remote_code=False`.

## Landmarks per model

| model | blocks | peak p | flip p+1 | ablation band | readout | peak hidden level |
|---|---|---|---|---|---|---|
| Qwen2.5-1.5B | 28 | 15 | 16 | 16-18 | 18 | 16 |
| Qwen2.5-3B | 36 | 16 | 17 | 17-19 | 19 | 17 |
| Llama-3.2-1B | 16 | 7 | 8 | 8-10 | 10 | 8 |
| Llama-3.2-3B | 28 | 9 | 10 | 10-12 | 12 | 10 |

Dataset: `NeelNanda/counterfact-tracing`, revision `c945b08...` (pinned
in the paper); TruthfulQA generation config, revision `741b827...`.

## Part I (Qwen2.5-1.5B)

```
python src/behav_check.py --model Qwen/Qwen2.5-1.5B --n 200 --seed 0
python src/truth_probe.py signal --dataset builtin --with-2d --baseline --perm 200
python src/truth_probe.py signal --dataset counterfact --max-pairs 250 --baseline --perm 200
python src/truth_probe.py polarity
python src/truth_probe.py recovery
python src/truth_probe.py domino --dataset builtin --perm 100
python src/canvas.py --dataset counterfact --layer 16 --out counterfact
python src/anatomy.py --dataset counterfact --max-pairs 250 --perm 100
python src/ablation.py --dataset counterfact --max-pairs 250 --band-start 16 --band-end 18 --readout 18
python src/axis_norm_check.py --model Qwen/Qwen2.5-1.5B --layer 16
```

## Part II (repeat with each model and its landmarks)

```
python src/behav_check.py --model meta-llama/Llama-3.2-3B --n 200 --seed 0
python src/flip_consolidate.py --model meta-llama/Llama-3.2-3B --axis-block 9 --flip-layer 10 --scan-start 5 --scan-end 15
python src/axis_provenance.py --model meta-llama/Llama-3.2-3B --peak 9 --scan-start 5 --scan-end 15
python src/ffn_erosion.py ablate --model meta-llama/Llama-3.2-3B --band-start 10 --band-end 12 --readout 12
python src/swiglu.py attrib --model meta-llama/Llama-3.2-3B --axis-block 9 --scan-start 6 --scan-end 13
python src/circuits.py --model meta-llama/Llama-3.2-3B --dtype bfloat16 --peak 9 --flip 10
python src/categories.py --model meta-llama/Llama-3.2-3B --peak 9 --write-layer 10 --k-relations 8
python src/arrangement_law.py
```

## Stress tests and the consensus gauge

```
# export dictionaries at any scale (this produced Data/dictionaries/)
python src/crea_dizionario.py --models Qwen/Qwen2.5-3B --k-relations 33 --pairs-per-relation 60 --seed 0 --out-dir dizionari

# consensus sign gauge (writes <bundle>_gauge.json, original untouched)
python src/reorient_gauge.py Data/dictionaries/Qwen2.5-3B/*.pt --mode eigen

# arrangement stress test between two gauged dictionaries
python src/arrangement_stress_test.py --a <A_gauge.json> --b <B_gauge.json>

# behavioral know-rate per relation (this produced Data/know_rates/)
python src/know_rate_per_relation.py --model Qwen/Qwen2.5-3B
python src/know_rate_per_relation.py --model meta-llama/Llama-3.2-3B
```

##Note

on `arrangement_law.py`: it carries the canonical K=8 matrices
transcribed in its source, verified cell-by-cell against fresh
`categories.py` runs; regeneration commands are in its header.

The code was generated using an LLM, subject to my full decision-making and review,
and to refine specific terminology necessitated by the translation from Italian to English.


