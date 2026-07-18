# TruthProbe

**The Anatomy of a Truth Direction**: knowledge-dependent dimensionality, a
relational law, and a convergent category geometry in small language models.

A training-free truth axis, read from the SVD of hidden-state minimal
pairs: label-free up to one sign bit, evaluated held-out against
permutation nulls, at a reading cost of two dot products per token.
The axis is followed across four base models and two families, opened
into a semantically signed mixture of per-category directions, and
stress-tested at three scales under a declared spectral sign gauge.

- **Paper (canonical):** [`Paper/The Anatomy of a Truth Direction.pdf`](Paper/)
  (Part I: the axis on one model; Part II: mechanics and geometry across
  two families; stress tests and the knowledge-gated arrangement law)
- **Project page:** the interactive observatory is served from this
  repository (`index.html`)
- **Code archive:** Zenodo DOI
  [10.5281/zenodo.20938285](https://doi.org/10.5281/zenodo.20938285)

## Layout

| path | content |
|---|---|
| `Paper/` | the consolidated manuscript (tex + pdf) and its figures |
| `src/` | every measurement tool: one command per reported number, no automatic verdicts |
| `Data/dictionaries/` | exported truth dictionaries per model, at three scales (K=8 n=60 with six extra seeds, K=8 n=888, K=33 n=60), with consensus-gauge signed matrices (`*_gauge.json`) |
| `Data/know_rates/` | behavioral per-relation know-rates (greedy string-match lower bound) |
| `Data/predictions/` | predictions registered on file before the runs, confirmed and falsified alike |
| `drafts/` | the paper trail: earlier versions, kept append-only |

Every exported artifact carries its identity (model, K, pairs, seed,
dataset revision) inside the file and in the filename; provenance tools
(`src/inventario_dizionari.py`, `src/confronta_json.py`,
`src/identikit_pt.py`) verify content by hash and cosine fingerprint.
Names can lie; axes cannot.

## Reproduce

See [`REPRODUCING.md`](REPRODUCING.md): golden rules, per-model landmark
table, and the exact command for every number in the paper.

```
pip install torch transformers datasets
python src/truth_probe.py signal --dataset builtin --with-2d --baseline --perm 200
```

## Note

The code was generated using an LLM, subject to my full decision-making and review, and to refine specific terminology necessitated by the translation from Italian to English.

## Cite

Francesco Karim Vicidomini, *The Anatomy of a Truth Direction*, 2026.
Code and data: doi:10.5281/zenodo.20938285. License: CC BY 4.0.
