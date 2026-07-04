# -*- coding: utf-8 -*-
"""Synthetic sanity tests for swiglu.py (no model needed)."""
import torch
import truth_probe as T
from swiglu import gate_value_split, cells_framing, build_framing_items, FRAMING_QUADS

torch.manual_seed(0)

# --- 1. exactness of the split identity, random tensors ---
d_i = 512
for _ in range(5):
    g_t, g_f = torch.randn(d_i), torch.randn(d_i)
    u_t, u_f = torch.randn(d_i), torch.randn(d_i)
    w = torch.randn(d_i)
    gate, value, total = gate_value_split(g_t, u_t, g_f, u_f, w)
    assert abs(float(gate + value - total)) < 1e-4 * max(1.0, abs(float(total))), \
        "split identity violated"
print("1. split identity exact on random tensors: OK")

# --- 2. planted attribution: gate-driven vs value-driven class signal ---
# gate-driven: u identical within pair, g differs along a neuron set S
S = torch.zeros(d_i); S[:10] = 1.0
w = S.clone()
u = torch.randn(d_i)
g_base = torch.rand(d_i)
g_t = g_base + 0.5 * S; g_f = g_base - 0.5 * S
gate, value, total = gate_value_split(g_t, u, g_f, u, w)
assert abs(float(value)) < 1e-5 and abs(float(gate - total)) < 1e-5
print(f"2a. gate-driven signal -> gate term {float(gate):+.3f}, value term {float(value):+.3f}: OK")
# value-driven: g identical, u differs
g = torch.rand(d_i)
u_t = u + 0.5 * S; u_f = u - 0.5 * S
gate, value, total = gate_value_split(g, u_t, g, u_f, w)
assert abs(float(gate)) < 1e-5 and abs(float(value - total)) < 1e-5
print(f"2b. value-driven signal -> gate term {float(gate):+.3f}, value term {float(value):+.3f}: OK")

# --- 3. framing cells + planted framing factor ---
items, fact, is_true, is_prud = build_framing_items()
n_facts = int(fact.max()) + 1
assert len(items) == 4 * n_facts == 4 * len(FRAMING_QUADS)
c = cells_framing(fact, is_true, is_prud, list(range(n_facts)))
assert len(c["MT"]) == len(c["MF"]) == len(c["PT"]) == len(c["PF"]) == n_facts
# no index overlaps
allidx = torch.cat([c["MT"], c["MF"], c["PT"], c["PF"]])
assert len(set(allidx.tolist())) == 4 * n_facts
print(f"3a. framing cells: {n_facts} facts x 4 cells, disjoint: OK")

# plant: truth axis e0, framing axis e1 rotates the truth direction for prudential
d = 32
N = len(items)
H = torch.randn(N, d) * 0.4
for i in range(N):
    tr = 1.0 if int(is_true[i]) else -1.0
    if int(is_prud[i]) == 0:
        H[i, 0] += 2.0 * tr                    # moral truth on e0
    else:
        H[i, 0] += -0.3 * tr; H[i, 1] += 2.0 * tr  # prudential truth ~orthogonal, slight anti
    H[i, 2] += 3.0 * int(is_prud[i])           # framing offset on e2 (class-blind)
tM = T.unit(H[c["MT"]].mean(0) - H[c["MF"]].mean(0))
tPd = T.unit(H[c["PT"]].mean(0) - H[c["PF"]].mean(0))
cos = float(torch.dot(tM, tPd))
auc_mm = T.auc_score(H[torch.cat([c["MT"], c["MF"]])] @ tM,
                     is_true[torch.cat([c["MT"], c["MF"]])].long())
auc_mp = T.auc_score(H[torch.cat([c["PT"], c["PF"]])] @ tM,
                     is_true[torch.cat([c["PT"], c["PF"]])].long())
print(f"3b. planted rotation: cos(tM,tP) {cos:.3f} (<1), M->M {auc_mm:.3f} high, "
      f"M->P {auc_mp:.3f} lower: ", end="")
assert cos < 0.5 and auc_mm > 0.95 and auc_mp < 0.8
print("OK")

print("ALL SWIGLU SANITY CHECKS PASSED")
