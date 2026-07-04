# -*- coding: utf-8 -*-
"""Synthetic sanity test for ffn_erosion helpers (no model needed).

Construct states with a planted truth axis u: true at +2u, false at -2u, plus noise.
Two synthetic 'FFN contributions':
  f_blind = big class-independent vector w (orthogonal to u)  -> gap ~ 0, d' ~ 0
  f_anti  = -1.5u * class_sign (writes against the truth axis) -> gap < 0, |d'| large
Check:
  1. heldout_contrib_gap signs/sizes come out as designed.
  2. heldout_auc_refit on clean states is high; adding f_anti to states lowers it;
     adding f_blind (orthogonal) leaves fixed-axis AUC ~unchanged.
  3. heldout_auc_fixed(intact, intact) == heldout_auc_refit(intact).
"""
import torch
import truth_probe as T
from ffn_erosion import heldout_contrib_gap, heldout_auc_refit, heldout_auc_fixed

torch.manual_seed(0)
d, n_pairs = 64, 40
u = torch.zeros(d); u[0] = 1.0
w = torch.zeros(d); w[1] = 1.0

pairs_states, pidx, items = [], [], []
H = []
for p in range(n_pairs):
    base = torch.randn(d) * 0.5              # shared topic
    ht = base + 2.0 * u + torch.randn(d) * 0.3
    hf = base - 2.0 * u + torch.randn(d) * 0.3
    it = len(H); H.append(ht)
    iff = len(H); H.append(hf)
    pidx.append((it, iff))
H = torch.stack(H, 0)
y = torch.tensor([1, 0] * n_pairs)

# synthetic contributions
F_blind = w.repeat(len(H), 1) * 5.0 + torch.randn(len(H), d) * 0.1
sign = torch.tensor([1.0 if l == 1 else -1.0 for l in y]).unsqueeze(1)
F_anti = (-1.5 * u).unsqueeze(0) * sign + torch.randn(len(H), d) * 0.1

g_b, d_b = heldout_contrib_gap(H, F_blind, pidx, 5, 0)
g_a, d_a = heldout_contrib_gap(H, F_anti, pidx, 5, 0)
print(f"blind writer : gap {g_b:+.3f}  d' {d_b:+.2f}   (expect ~0)")
print(f"anti  writer : gap {g_a:+.3f}  d' {d_a:+.2f}   (expect strongly negative)")
assert abs(d_b) < 0.5 and d_a < -3, "contrib gap logic broken"

auc_clean = heldout_auc_refit(H, pidx, 5, 0)
auc_fixed_self = heldout_auc_fixed(H, H, pidx, 5, 0)
H_anti = H + F_anti
H_blind = H + F_blind
auc_anti_refit = heldout_auc_refit(H_anti, pidx, 5, 0)
auc_anti_fixed = heldout_auc_fixed(H, H_anti, pidx, 5, 0)
auc_blind_fixed = heldout_auc_fixed(H, H_blind, pidx, 5, 0)
print(f"clean refit {auc_clean:.3f} | fixed(self) {auc_fixed_self:.3f} (must match)")
print(f"anti-writer added : refit {auc_anti_refit:.3f}  fixed {auc_anti_fixed:.3f} (both should drop)")
print(f"blind-writer added: fixed {auc_blind_fixed:.3f} (orthogonal -> should stay high)")
assert abs(auc_clean - auc_fixed_self) < 1e-6
assert auc_anti_fixed < auc_clean - 0.1
assert auc_blind_fixed > auc_clean - 0.05
print("ALL SANITY CHECKS PASSED")
