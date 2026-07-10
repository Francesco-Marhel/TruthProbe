# -*- coding: utf-8 -*-
"""
circuits.py  --  Static circuit anatomy: does the routing->writing crossover of
the attention circuits coincide with the truth peak and the FFN flip?

Background. Per head, attention factorizes into two GAUGE-INVARIANT circuits
(Elhage et al. 2021):
    QK circuit  M_qk = W_q^T W_k   (routing: which positions talk)
    OV circuit  M_ov = W_o  W_v    (writing: what gets moved)
Individual norms ||W_q||, ||W_v|| are gauge-dependent (rescale W_q by a and W_k
by 1/a: same function); the PRODUCTS are not. Weight decay tends to balance the
coupled norms during training, which is why raw V/Q ratios can still correlate
with depth -- but the products are the clean object, so we measure those.

Registered prediction (write it down BEFORE running): if the 'relay' story is
right, the layer where ||OV|| overtakes ||QK|| (suitably normalized) should sit
near the measured landmarks of this repo: truth peak at block 15, FFN value-flip
at 16, erosion band 16-18, evaporation from 19. Crossover far from 15-16
falsifies the link between the static scaling law and our causal anatomy.

GQA caveat handled explicitly: Qwen2.5-1.5B has 12 query heads but only 2 KV
heads; each group of 6 Q-heads shares one K and one V. All per-head numbers
below pair Q-head h with KV group h // (n_q / n_kv). RoPE caveat: W_q^T W_k is
the pre-rotary bilinear form (standard practice; position-dependent rotation
sits between them).

Weights only -- no forward pass, runs in seconds, CPU is fine.

    python circuits.py            # full table + crossover + per-head map
    python circuits.py --per-head # add the per-head (12 x 28) breakdown
"""
import argparse
import torch


def head_slices(W, n_heads, head_dim, dim=0):
    """Split a projection weight [out, in] into per-head blocks along `dim`."""
    return [W.narrow(dim, h * head_dim, head_dim) for h in range(n_heads)]


def qk_norm(Wq_h, Wk_g):
    """||W_q^T W_k||_F for one (Q-head, KV-group) pair. Wq_h, Wk_g: [head_dim, d]."""
    return float((Wq_h.T @ Wk_g).norm())


def ov_norm(Wo_h, Wv_g):
    """||W_o W_v||_F for one pair. Wo_h: [d, head_dim], Wv_g: [head_dim, d]."""
    return float((Wo_h @ Wv_g).norm())


def layer_circuits(layer, n_q, n_kv, head_dim):
    """Per-Q-head QK and OV circuit norms for one attention block (GQA-aware)."""
    attn = layer.self_attn
    Wq = attn.q_proj.weight.detach().float()          # [n_q*hd, d]
    Wk = attn.k_proj.weight.detach().float()          # [n_kv*hd, d]
    Wv = attn.v_proj.weight.detach().float()          # [n_kv*hd, d]
    Wo = attn.o_proj.weight.detach().float()          # [d, n_q*hd]
    q_h = head_slices(Wq, n_q, head_dim, dim=0)
    k_g = head_slices(Wk, n_kv, head_dim, dim=0)
    v_g = head_slices(Wv, n_kv, head_dim, dim=0)
    o_h = head_slices(Wo, n_q, head_dim, dim=1)
    group = n_q // n_kv
    qk = [qk_norm(q_h[h], k_g[h // group]) for h in range(n_q)]
    ov = [ov_norm(o_h[h], v_g[h // group]) for h in range(n_q)]
    return qk, ov


def gauge_invariance_selftest():
    """Rescale Wq by alpha, Wk by 1/alpha: the QK norm must NOT move.
    Same for Wv vs Wo. If this fails, the computation is wrong -- abort."""
    torch.manual_seed(0)
    d, hd = 64, 16
    Wq = torch.randn(hd, d); Wk = torch.randn(hd, d)
    Wv = torch.randn(hd, d); Wo = torch.randn(d, hd)
    a = 7.3
    e1 = abs(qk_norm(Wq, Wk) - qk_norm(a * Wq, Wk / a)) / qk_norm(Wq, Wk)
    e2 = abs(ov_norm(Wo, Wv) - ov_norm(Wo / a, a * Wv)) / ov_norm(Wo, Wv)
    # and confirm the RAW norms DO move (i.e. the gauge is real, not trivial)
    raw_moves = abs(float((a * Wq).norm()) - float(Wq.norm())) > 1.0
    assert e1 < 1e-5 and e2 < 1e-5 and raw_moves, "gauge self-test failed"
    return e1, e2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
    ap.add_argument("--dtype", default="float32",
                    choices=["float32", "bfloat16", "float16"],
                    help="load dtype (bf16 halves memory for big models; per-head "
                         "slices are upcast to fp32 before the norm, so precision is safe)")
    ap.add_argument("--per-head", action="store_true",
                    help="print the per-head (n_q x L) breakdown too")
    ap.add_argument("--peak", type=int, default=None,
                    help="truth-peak block of THIS model (annotations off if omitted)")
    ap.add_argument("--flip", type=int, default=None,
                    help="flip block of THIS model (default: peak+1 when --peak given)")
    a = ap.parse_args()

    e1, e2 = gauge_invariance_selftest()
    print(f"[gauge self-test] QK invariance err {e1:.1e}, OV invariance err {e2:.1e}  OK")

    print(f"[model] loading {a.model} (weights only; CPU is fine)")
    from transformers import AutoConfig, AutoModelForCausalLM
    cfg = AutoConfig.from_pretrained(a.model, trust_remote_code=False)
    dt = {"float32": torch.float32, "bfloat16": torch.bfloat16,
          "float16": torch.float16}[a.dtype]
    model = AutoModelForCausalLM.from_pretrained(
        a.model, dtype=dt, use_safetensors=True, trust_remote_code=False)
    model.eval()

    n_q = cfg.num_attention_heads
    n_kv = cfg.num_key_value_heads
    head_dim = getattr(cfg, "head_dim", cfg.hidden_size // n_q)
    L = cfg.num_hidden_layers
    print(f"[config] layers {L}  d_model {cfg.hidden_size}  Q-heads {n_q}  "
          f"KV-heads {n_kv} (GQA group = {n_q // n_kv})  head_dim {head_dim}")
    if n_kv < n_q:
        print(f"[GQA] per-head numbers pair Q-head h with KV group h // {n_q // n_kv}.")

    QK, OV = [], []      # [L][n_q]
    for i, layer in enumerate(model.model.layers):
        qk, ov = layer_circuits(layer, n_q, n_kv, head_dim)
        QK.append(qk); OV.append(ov)
        print(f"\r[circuits] layer {i+1}/{L}", end="", flush=True)
    print()

    qk_sum = [sum(r) for r in QK]
    ov_sum = [sum(r) for r in OV]
    # normalize each series by its own median so the ratio is scale-free
    def norml(xs):
        m = sorted(xs)[len(xs) // 2]
        return [x / m for x in xs]
    qk_n, ov_n = norml(qk_sum), norml(ov_sum)
    ratio = [o / q for o, q in zip(ov_n, qk_n)]

    peak = a.peak
    flip = a.flip if a.flip is not None else (peak + 1 if peak is not None else None)
    landmarks = {}
    if peak is not None:
        landmarks[peak] = "TRUTH PEAK"
        landmarks[flip] = "FFN value flip"
    print(f"\n{'layer':>5} | {'||QK|| sum':>10} {'||OV|| sum':>10} | "
          f"{'QK (norm)':>9} {'OV (norm)':>9} | {'OV/QK':>6}")
    print("-" * 72)
    for i in range(L):
        mark = landmarks.get(i, "")
        print(f"{i:>5} | {qk_sum[i]:>10.1f} {ov_sum[i]:>10.1f} | "
              f"{qk_n[i]:>9.2f} {ov_n[i]:>9.2f} | {ratio[i]:>6.2f}  {mark}")

    # crossover: first layer from which OV/QK stays above 1 for >=3 consecutive layers
    cross = None
    for i in range(L - 2):
        if all(ratio[j] > 1.0 for j in range(i, min(i + 3, L))):
            cross = i; break
    print("\n=== crossover (no auto-verdict; read against YOUR landmarks) ===")
    if cross is None:
        print("  no sustained OV>QK crossover (>=3 consecutive layers above 1).")
    else:
        print(f"  sustained OV/QK > 1 from layer {cross}")
        if peak is not None:
            dist = min(abs(cross - peak), abs(cross - flip))
            print(f"  distance from peak {peak} / flip {flip}: {dist}   "
                  f"relative depth: crossover {cross}/{L} = {cross/L:.2f}, "
                  f"peak {peak}/{L} = {peak/L:.2f}")

    if a.per_head:
        print("\n[per-head OV/QK ratio, rows = heads, cols = layers]")
        for h in range(n_q):
            row = [OV[i][h] / max(QK[i][h], 1e-8) for i in range(L)]
            print(f"  h{h:02d} " + " ".join(f"{x:5.1f}" for x in row))
        print("  (heads sharing a KV group differ only through their Wq/Wo slices)")


if __name__ == "__main__":
    main()
