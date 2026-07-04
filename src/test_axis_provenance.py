# -*- coding: utf-8 -*-
"""Synthetic validations for axis_provenance.py (no model needed).

Test 1 -- ground-truth pinning: a GLOBAL planted axis present at every block,
with an FFN writer that is pro-truth before layer k and anti-truth from k on.
The measured sign transition must sit at k under EVERY ruler fit block
(if it tracked the ruler instead, the instrument would manufacture the law).

Test 2 -- PRE/POST collapse-ratio discrimination:
  scenario A (CIRCULAR): the writer is class-signed along directions
    orthogonal to the true axis, folded into each block's output. The POST
    ruler (fit on a state containing the write) reads it as pro; the PRE
    ruler must expose it:  |d_pre| < 0.25 * d_post.
  scenario B (GENUINE): the writer is class-signed along the existing axis.
    Alignment must survive the PRE ruler:  d_pre > 0.60 * d_post.

Run:  python test_axis_provenance.py
"""
import torch
import truth_probe as T
from flip_consolidate import axis_per_fold, gap_on_axis

torch.manual_seed(0)


def test_ground_truth_pinning():
    d, n_pairs, Lb = 64, 40, 10
    u = torch.zeros(d); u[0] = 1.0
    k = 6
    H_resid = torch.zeros(2 * n_pairs, Lb + 1, d)
    H_ffn = torch.zeros(2 * n_pairs, Lb, d)
    pidx = []
    for p in range(n_pairs):
        base = torch.randn(d) * 0.5
        it, iff = 2 * p, 2 * p + 1
        pidx.append((it, iff))
        for l in range(Lb + 1):
            dist = torch.randn(d) * 0.4     # per-block distortion: rulers differ
            H_resid[it, l] = base + 2 * u + dist + torch.randn(d) * 0.3
            H_resid[iff, l] = base - 2 * u + dist + torch.randn(d) * 0.3
        for l in range(Lb):
            s = +0.8 if l < k else -0.8
            H_ffn[it, l] = s * u + torch.randn(d) * 0.1
            H_ffn[iff, l] = -s * u + torch.randn(d) * 0.1
    scan = list(range(2, 10))
    for b in [2, 4, 6, 8]:
        axes = axis_per_fold(H_resid[:, b + 1, :], pidx, 5, 0)
        ds = {L: gap_on_axis(H_ffn[:, L, :], pidx, axes)[1] for L in scan}
        t = None
        for i, L in enumerate(scan[:-1]):
            if ds[L] < -0.10 and ds[scan[i + 1]] < 0:
                t = L; break
        print(f"  ruler @block {b}: transition at {t} (ground truth {k})")
        assert t == k, "transition must be pinned at the true writer flip"
    print("test 1 (ground-truth pinning under every ruler): PASSED")


def _build(scenario, d=96, n_pairs=40, L=8):
    u = torch.zeros(d); u[0] = 1.0
    H_resid = torch.zeros(2 * n_pairs, L + 1, d)
    H_attn = torch.zeros(2 * n_pairs, L, d)
    H_ffn = torch.zeros(2 * n_pairs, L, d)
    pidx, rs, basis = [], [], [u]
    for l in range(L):
        r = torch.randn(d)
        for v in basis:                     # Gram-Schmidt vs u and previous r's
            r = r - (r @ v) * v
        r = r / r.norm(); rs.append(r); basis.append(r)
    for p in range(n_pairs):
        base = torch.randn(d) * 0.5
        it, iff = 2 * p, 2 * p + 1
        pidx.append((it, iff))
        H_resid[it, 0] = base + 2 * u + torch.randn(d) * 0.3
        H_resid[iff, 0] = base - 2 * u + torch.randn(d) * 0.3
        for l in range(L):
            a_t = torch.randn(d) * 0.1; a_f = torch.randn(d) * 0.1
            dirv = rs[l] if scenario == "A" else u
            f_t = +1.0 * dirv + torch.randn(d) * 0.3
            f_f = -1.0 * dirv + torch.randn(d) * 0.3
            H_attn[it, l], H_attn[iff, l] = a_t, a_f
            H_ffn[it, l], H_ffn[iff, l] = f_t, f_f
            H_resid[it, l + 1] = H_resid[it, l] + a_t + f_t
            H_resid[iff, l + 1] = H_resid[iff, l] + a_f + f_f
    return H_resid, H_attn, H_ffn, pidx


def test_collapse_ratio():
    for scen in ("A", "B"):
        H_resid, H_attn, H_ffn, pidx = _build(scen)
        b = 4
        post = axis_per_fold(H_resid[:, b + 1, :], pidx, 5, 0)
        pre = axis_per_fold(H_resid[:, b, :] + H_attn[:, b, :], pidx, 5, 0)
        _, d_post = gap_on_axis(H_ffn[:, b, :], pidx, post)
        _, d_pre = gap_on_axis(H_ffn[:, b, :], pidx, pre)
        print(f"  scenario {scen}: d'(b) POST {d_post:+.2f}  PRE {d_pre:+.2f}  "
              f"ratio {d_pre / d_post:+.2f}")
        if scen == "A":
            assert abs(d_pre) < 0.25 * abs(d_post), "circularity not exposed"
        else:
            assert d_pre > 0.60 * d_post, "genuine alignment not preserved"
    print("test 2 (PRE/POST collapse-ratio discrimination): PASSED")


if __name__ == "__main__":
    test_ground_truth_pinning()
    test_collapse_ratio()
    print("ALL AXIS-PROVENANCE SYNTHETIC VALIDATIONS PASSED")
