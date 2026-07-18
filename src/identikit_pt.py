"""
identikit_pt.py -- identify an anonymous .pt bundle by its cosine fingerprint.

Loads the orphan's axes, builds its signed cosine matrix, and compares it
against the cos_peak of every candidate JSON you pass: exact match, and
match up to per-axis sign flips (in case the orphan predates the current
orientation). The content identifies itself; names can lie, axes cannot.

Usage:
  python identikit_pt.py archivio/truthdict_model-x_K8_nx_seedx_fab5c0c4.pt ^
      archivio/truthdict_Llama-3.2-3B_K8_n60_seed0.json ^
      archivio/truthdict_Qwen2.5-3B_K8_n60_seed0.json ^
      archivio/truthdict_Llama-3.2-3B_K8_n888_seed0.json ^
      archivio/truthdict_Qwen2.5-3B_K8_n888_seed0.json
"""
import json, sys
import torch

def unit(v): return v / v.norm().clamp_min(1e-8)

orphan = sys.argv[1]
cands = sys.argv[2:]
d = torch.load(orphan, map_location="cpu")
if "axes" not in d:
    sys.exit(f"{orphan}: no axes inside, cannot fingerprint")
A = torch.stack([unit(a) for a in d["axes"].float()])
M = A @ A.T
cats_o = list(d.get("cats", []))
K = M.shape[0]
print(f"orphan: {orphan}   K={K}   cats={cats_o or 'absent'}\n")

for c in cands:
    j = json.load(open(c))
    cj = j.get("cats", [])
    if len(cj) != K:
        print(f"{c}\n   K differs ({len(cj)}): not this one\n"); continue
    R = torch.tensor(j["cos_peak"], dtype=torch.float32)
    raw = float((M - R).abs().max())
    # match up to per-axis signs: greedy flip to minimize distance
    S = torch.ones(K)
    improved = True
    while improved:
        improved = False
        for i in range(K):
            cur = float(((S[:, None] * S[None, :] * M) - R).abs().max())
            S[i] *= -1
            new = float(((S[:, None] * S[None, :] * M) - R).abs().max())
            if new < cur - 1e-9: improved = True
            else: S[i] *= -1
    rep = float(((S[:, None] * S[None, :] * M) - R).abs().max())
    flips = [cj[i] for i in range(K) if S[i] < 0]
    verdict = ("MATCH (identical)" if raw < 5e-3 else
               f"MATCH up to sign flips {flips}" if rep < 5e-3 else
               "no")
    print(f"{c}\n   max|delta| raw = {raw:.4f}   after sign repair = {rep:.4f}   -> {verdict}\n")
