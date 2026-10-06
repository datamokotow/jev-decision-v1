"""Validate a decision JSONL file and print a profile (counts per type/task, option counts,
token-length percentiles if a tokenizer is given). Run this on corp data before training.

    python scripts/validate_jsonl.py corp_train.jsonl [--tokenizer /models/bge-m3 --max_length 2048]
"""
import argparse
import collections
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from jev.data import encode_row, read_jsonl  # noqa: E402


def main():
    a = argparse.ArgumentParser()
    a.add_argument("path")
    a.add_argument("--tokenizer")
    a.add_argument("--max_length", type=int, default=2048)
    a.add_argument("--head_length", type=int, default=512)
    args = a.parse_args()
    rows = read_jsonl(args.path, require_target=False)
    print(f"{len(rows)} valid rows")
    print("types:", dict(collections.Counter(r.get("type", "choice") for r in rows)))
    print("tasks:", dict(collections.Counter(r.get("task", "-") for r in rows)))
    print("options per row:", dict(sorted(collections.Counter(len(r["options"]) for r in rows).items())))
    labelled = [r for r in rows if "target" in r]
    if labelled:
        pos = collections.Counter(r["target"] for r in labelled)
        top = max(pos.values()) / len(labelled)
        print(f"target position share (max {top:.0%}):", dict(sorted(pos.items())))
        if top > 0.6:
            print("WARNING: targets concentrated on one position - shuffle options, or the model learns position.")
    if args.tokenizer:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.tokenizer)
        enc = [encode_row(tok, r, args.max_length, args.head_length) for r in rows]
        lens = sorted(len(e["ids"]) for e in enc)
        pct = lambda q: lens[min(len(lens) - 1, int(q * len(lens)))]
        print(f"tokens p50={pct(.5)} p90={pct(.9)} p99={pct(.99)} max={lens[-1]}")
        print(f"state truncated in {sum(e['truncated'] for e in enc) / len(enc):.1%} of rows (max_length={args.max_length})")


if __name__ == "__main__":
    main()
