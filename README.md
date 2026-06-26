# How Much of Truth Fits on a Single Axis?
### Knowledge-Dependent Dimensionality and Unsupervised Polarity Recovery in Large Language Model Representations

This repository contains the official implementation and paper for the unsupervised geometric truth probe on LLM representations.

## Core Findings
1. Unsupervised Truth Signal: The geometric axis extracted via SVD carries a real truth signal, reaching a held-out AUC of up to 0.938 on clean minimal pairs.
2. Knowledge-Dependent Dimensionality: The gap between the unsupervised axis and a full linear probe scales with how well the model knows the fact.
3. Unsupervised Polarity Recovery: Unsupervised SVD on mixed-polarity data recovers the supervised polarity direction at a 0.959 cosine similarity, without seeing any polarity labels.

## Repository Structure
* truth_probe.py: Python script containing the model execution and testing pipeline.
* paper.pdf: PDF document with the full theoretical framework.
* LICENSE: Apache License 2.0.

## Reproducibility
To replicate the results reported in the paper, install the required dependencies and run the corresponding subcommands.

Dependencies installation:
```bash
pip install torch transformers datasets
```

To reproduce Table 1 (Clean Minimal Pairs):
```bash
python truth_probe.py signal --dataset builtin --with-2d --baseline --perm 200
```

To reproduce Table 1 (CounterFact / TruthfulQA Mix):
```bash
python truth_probe.py signal --dataset mix --max-pairs 250 --with-2d --baseline --perm 200
```

To reproduce Table 2 (Polarity Analysis):
```bash
python truth_probe.py polarity
```

To reproduce Table 2 (Unsupervised SVD Polarity Recovery):
```bash
python truth_probe.py recovery
```

## License and Citation
The code in this repository is licensed under the Apache License 2.0. If you use this theory or implementation in your research, please cite the original author:

* Author: Francesco Karim Vicidomini
* Date: June 2026

* ## Citation

If you use this theoretical framework or implementation in your research, please use the following BibTeX entry:

```bibtex
@misc{vicidomini2026truth,
  author       = {Vicidomini, Francesco Karim},
  title        = {How Much of Truth Fits on a Single Axis? Knowledge-Dependent Dimensionality and Unsupervised Polarity Recovery in Large Language Model Representations},
  year         = {2026},
  doi          = {10.5281/zenodo.20938286},
  url          = {https://github.com}
}
```


