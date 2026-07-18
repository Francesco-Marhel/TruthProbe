"""
know_rate_per_relation.py -- behavioral know-rate, per CounterFact relation.

For each relation, greedy open-ended completion on up to N prompts and the
exact criterion of behav_check.py: known if the completion starts with the
true target or contains it preceded by a space, case-insensitive, within
8 new tokens. Measurement only: prints the table, saves a JSON, no verdict.
The relation list defaults to the 33 categories of the K33 dictionaries,
so the estimate matches the geometry's categories by construction.

Usage:
  python know_rate_per_relation.py --model Qwen/Qwen2.5-3B
  python know_rate_per_relation.py --model meta-llama/Llama-3.2-3B
"""
import argparse, json, random
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

K33_RELATIONS = ["P101","P103","P106","P108","P127","P1303","P131","P136",
                 "P138","P140","P1412","P159","P17","P176","P178","P19",
                 "P190","P20","P27","P276","P30","P36","P364","P37","P39",
                 "P407","P413","P449","P463","P495","P641","P740","P937"]
REV = "c945b082ca08d0a8f3ba227fb78404a09614c36e"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--per-relation", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--rev-counterfact", default=REV)
    ap.add_argument("--relations", nargs="*", default=K33_RELATIONS)
    a = ap.parse_args()

    ds = load_dataset("NeelNanda/counterfact-tracing", split="train",
                      revision=a.rev_counterfact)
    cols = ds.column_names
    rel_col = next(c for c in ("relation_id", "relation", "predicate_id") if c in cols)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(a.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        a.model, torch_dtype=torch.bfloat16, use_safetensors=True,
        trust_remote_code=False).to(dev).eval()

    by_rel = {}
    for row in ds:
        r = str(row[rel_col])
        if r in a.relations:
            by_rel.setdefault(r, []).append((row["prompt"], row["target_true"]))

    rng = random.Random(a.seed)
    out = {"model": a.model, "revision": a.rev_counterfact, "seed": a.seed,
           "per_relation_max": a.per_relation, "criterion": "behav_check.py",
           "per_relation": {}}
    print(f"{'relation':8s} {'n':>4s} {'known':>6s} {'know-rate':>10s}")
    for rel in a.relations:
        rows = by_rel.get(rel, [])
        rng.shuffle(rows)
        rows = rows[: a.per_relation]
        known = 0
        for i in range(0, len(rows), a.batch):
            chunk = rows[i:i + a.batch]
            enc = tok([p for p, _t in chunk], return_tensors="pt",
                      padding=True).to(dev)
            with torch.no_grad():
                gen = model.generate(**enc, max_new_tokens=8, do_sample=False,
                                     pad_token_id=tok.pad_token_id)
            for (p, t), seq in zip(chunk, gen):
                comp = tok.decode(seq[enc["input_ids"].shape[1]:],
                                  skip_special_tokens=True)
                cl, tl = comp.strip().lower(), t.strip().lower()
                if cl.startswith(tl) or (" " + tl) in comp.lower():
                    known += 1
        kr = known / len(rows) if rows else float("nan")
        out["per_relation"][rel] = {"n": len(rows), "known": known,
                                    "know_rate": round(kr, 4)}
        print(f"{rel:8s} {len(rows):>4d} {known:>6d} {kr:>10.3f}")

    dst = f"know_rate_{a.model.split('/')[-1]}_seed{a.seed}.json"
    with open(dst, "w") as f:
        json.dump(out, f, indent=1)
    print(f"\n[saved] {dst}")

if __name__ == "__main__":
    main()
