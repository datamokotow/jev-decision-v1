"""Build a small, leakage-checked training set from the public typed-decision datasets.

  train : hotchpotch/bekko-system-one-dataset-v0  (config laya__typed_decisions, split train) [+ extra configs]
  val   : same config, split validation
  test  : LocalLLaMA/typed-decisions               (config all, split test)   <- held-out evaluation

Rows whose state also appears in val/test are dropped from train. Run where huggingface.co is reachable.

    python scripts/build_typed_decisions.py --out data/typed --max_train 2500
"""
import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from jev.convert import bekko_row, holdout_by_state, state_key, typed_decisions_row  # noqa: E402


ERRORS = []


def safe(fn, row, *a, **k):
    """Convert one source row; schema surprises are counted and reported, not fatal."""
    try:
        return fn(row, *a, **k)
    except Exception as e:  # noqa: BLE001
        ERRORS.append(f"{type(e).__name__}: {e} | keys={list(row)[:8]}")
        return []


def write(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    from datasets import load_dataset
    a = argparse.ArgumentParser()
    a.add_argument("--out", default="data/typed")
    a.add_argument("--configs", nargs="+", default=["laya__typed_decisions"], help="bekko dataset configs")
    a.add_argument("--max_train", type=int, default=2500, help="max decision rows")
    a.add_argument("--max_val", type=int, default=400)
    a.add_argument("--max_test", type=int, default=600)
    a.add_argument("--seed", type=int, default=0)
    args = a.parse_args()
    rnd = random.Random(args.seed)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    test = []
    for r in load_dataset("LocalLLaMA/typed-decisions", "all", split="test"):
        test.extend(safe(typed_decisions_row, r, rnd))
    train, val, bekko_test = [], [], []
    for cfg in args.configs:
        for split, bucket in (("train", train), ("validation", val), ("test", bekko_test)):
            try:
                ds = load_dataset("hotchpotch/bekko-system-one-dataset-v0", cfg, split=split)
            except Exception as e:  # missing split/config: skip, keep going
                print(f"[skip] {cfg}/{split}: {type(e).__name__}: {str(e)[:100]}")
                continue
            n0 = len(bucket)
            for r in ds:
                bucket.extend(safe(bekko_row, r, source=cfg))
            print(f"[ok]   {cfg}/{split}: +{len(bucket) - n0} decisions")

    if ERRORS:
        print(f"WARNING: {len(ERRORS)} source rows failed to convert; first errors:")
        for e in ERRORS[:3]:
            print("   ", e[:300])
    if not train or not test:
        sys.exit(f"no usable data (train={len(train)}, test={len(test)}): schema probably changed - see errors above")
    held = {state_key(r) for r in test} | {state_key(r) for r in val} | {state_key(r) for r in bekko_test}
    before = len(train)
    train = [r for r in train if state_key(r) not in held]
    print(f"leakage filter: dropped {before - len(train)} train decisions whose state appears in val/test")
    if not val:  # this config ships no validation split: carve 10% of train *states* (never split one state)
        train, val = holdout_by_state(train, 0.10, rnd)
        print(f"no validation split in source: held out {len(val)} decisions by state")
    for rows in (train, val, test):
        rnd.shuffle(rows)
    train, val, test = train[:args.max_train], val[:args.max_val], test[:args.max_test]
    write(out / "train.jsonl", train); write(out / "val.jsonl", val); write(out / "test.jsonl", test)
    (out / "DATA_LICENSES.md").write_text(
        "- LocalLLaMA/typed-decisions: Apache-2.0\n"
        "- hotchpotch/bekko-system-one-dataset-v0: mixed per source subset; see the dataset card. "
        "laya__typed_decisions derives from LocalLLaMA/typed-decisions.\n")
    print(f"wrote train={len(train)} val={len(val)} test={len(test)} decisions to {out}")


if __name__ == "__main__":
    main()
