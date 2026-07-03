# -*- coding: utf-8 -*-
"""
behav_check.py  --  'Verify behavior before geometry' (companion paper, Sec. 9),
made quantitative: before porting the truth probe to a new model, measure how
much of the probed material the model actually KNOWS behaviorally.

Greedy-decodes CounterFact prompts (no context between prompts) and counts the
fraction where the true target appears at the start of the completion. Also
prints a handful of built-in sanity completions to eyeball.

The KNOWN fraction pre-registers the dimensionality-law expectation for the new
model: high fraction -> small axis-vs-probe gap expected; low fraction -> large
gap expected. Run this BEFORE truth_probe signal on the new model.

    python behav_check.py --model meta-llama/Llama-3.2-3B
"""
import argparse
import random
import torch
import truth_probe as T

SANITY_PROMPTS = [
    "The capital of France is",
    "The chemical symbol for gold is",
    "The Sun is a",
    "Water is made of hydrogen and",
    "Romeo and Juliet was written by",
]


def load_counterfact_prompts(n, seed, revision, local_file):
    if local_file:
        ds = T._open_local(local_file)
    else:
        from datasets import load_dataset
        ds = load_dataset("NeelNanda/counterfact-tracing", split="train", revision=revision)
    print(f"  [provenance] NeelNanda/counterfact-tracing @ {revision or 'latest'} ({len(ds)} rows)")
    idx = list(range(len(ds))); random.Random(seed).shuffle(idx)
    out, seen = [], set()
    for i in idx:
        ex = ds[i]
        prompt = str(ex["prompt"]).strip()
        tt = str(ex["target_true"]).strip()
        if not prompt or not tt or (prompt, tt) in seen:
            continue
        seen.add((prompt, tt))
        out.append((prompt, tt))
        if len(out) >= n:
            break
    return out


@torch.no_grad()
def greedy(model, tok, prompt, dev, max_new):
    ids = tok(prompt, return_tensors="pt").to(dev)
    out = model.generate(**ids, max_new_tokens=max_new, do_sample=False,
                         pad_token_id=tok.eos_token_id)
    return tok.decode(out[0, ids["input_ids"].shape[1]:], skip_special_tokens=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="meta-llama/Llama-3.2-3B")
    ap.add_argument("--n", type=int, default=40, help="CounterFact prompts to test")
    ap.add_argument("--max-new", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rev-counterfact", default="c945b082ca08d0a8f3ba227fb78404a09614c36e")
    ap.add_argument("--file-counterfact", default=None)
    a = ap.parse_args()

    dev, dt = T.resolve_device_dtype("auto", "auto")   # bf16 on cuda: generation only
    print("[task behav_check] behavior BEFORE geometry (Sec. 9)")
    print(f"[model] {a.model} on {dev} ({dt})")
    tok, model = T.load_model(a.model, dt, dev)

    print("\n--- sanity completions (eyeball) ---")
    for p in SANITY_PROMPTS:
        print(f"  {p!r} -> {greedy(model, tok, p, dev, a.max_new)!r}")

    print(f"\n--- CounterFact greedy check ({a.n} prompts) ---")
    prompts = load_counterfact_prompts(a.n, a.seed, a.rev_counterfact, a.file_counterfact)
    hits = 0
    for i, (prompt, tt) in enumerate(prompts):
        comp = greedy(model, tok, prompt, dev, a.max_new)
        ok = comp.strip().lower().startswith(tt.lower()) or (" " + tt.lower()) in comp.lower()
        hits += int(ok)
        mark = "KNOWN  " if ok else "unknown"
        print(f"  [{mark}] {prompt!r} -> {comp.strip()!r}   (true: {tt!r})")
        print(f"\r  {i+1}/{len(prompts)}  running KNOWN rate {hits/(i+1):.0%}", end="\n")
    frac = hits / len(prompts)
    print(f"\n=== KNOWN fraction: {frac:.0%} ({hits}/{len(prompts)}) ===")
    print("  pre-registered expectation for the dimensionality law on this model:")
    if frac >= 0.5:
        print("  HIGH knowledge -> expect a SMALL axis-vs-probe gap (known-fact regime,")
        print("  like Qwen2.5-3B's 0.049). A large gap would VIOLATE the law.")
    elif frac >= 0.25:
        print("  MID knowledge -> expect an intermediate gap. Note it before running signal.")
    else:
        print("  LOW knowledge -> expect a LARGE gap (unknown-fact regime, like the")
        print("  1.5B's ~0.20). A small gap would VIOLATE the law. Also: per Sec. 9,")
        print("  interpret the geometry cautiously -- the capability is mostly absent.")
    print("  (String-match on greedy decoding: a rough lower bound, not a strict label.)")


if __name__ == "__main__":
    main()
