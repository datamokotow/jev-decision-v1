"""Stream a large, mixed training pool out of the Bekko System One dataset (6.59M cases, 153 subsets).

Rows are streamed round-robin across subsets (so every subset is represented) and written straight to disk,
so memory stays flat even for ~1M decisions. Nothing is shuffled here: the trainer shuffles.

  train.jsonl       target ~N decisions; states in val/test are excluded (no leakage)
  val.jsonl         ~0.1% of states chosen by hash (from the same stream) - cheap, in-distribution
  test_bekko.jsonl  a few rows from each subset's own `test` split (broad, out-of-sample)
  test_typed.jsonl  LocalLLaMA/typed-decisions test split (the same held-out set as the Mac run)

    python scripts/build_bekko_large.py --out data/large --n_train 1000000
"""
import argparse
import hashlib
import itertools
import json
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from jev.convert import bekko_row, state_key, typed_decisions_row  # noqa: E402

DATASET = "hotchpotch/bekko-system-one-dataset-v0"
ERRORS = Counter()


def convert(fn, row, *a, **k):
    try:
        return fn(row, *a, **k)
    except Exception as e:  # noqa: BLE001  schema surprises are counted, never fatal
        ERRORS[f"{type(e).__name__}: {str(e)[:80]}"] += 1
        return []


def slim(r):
    r = dict(r); r.pop("group", None)
    return r


def main():
    from datasets import get_dataset_config_names, load_dataset
    a = argparse.ArgumentParser()
    a.add_argument("--out", default="data/large")
    a.add_argument("--n_train", type=int, default=1_000_000, help="target number of decisions")
    a.add_argument("--max_val", type=int, default=1000)
    a.add_argument("--test_rows_per_config", type=int, default=8)
    a.add_argument("--max_test", type=int, default=2000)
    a.add_argument("--configs", nargs="*", help="restrict to these subsets (default: all)")
    a.add_argument("--val_mod", type=int, default=1000, help="1 in N states goes to val")
    a.add_argument("--chunk", type=int, default=64, help="rows taken from one subset before moving to the next")
    a.add_argument("--max_configs", type=int, default=0, help="debug: only the first N subsets")
    args = a.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    configs = args.configs or get_dataset_config_names(DATASET)
    if args.max_configs:
        configs = configs[: args.max_configs]
    print(f"{len(configs)} subsets", flush=True)

    # ---- held-out sets first, so train can exclude their states ------------------------------
    held, test_typed, test_bekko = set(), [], []
    for r in load_dataset("LocalLLaMA/typed-decisions", "all", split="test"):
        test_typed.extend(convert(typed_decisions_row, r))
    held |= {state_key(r) for r in test_typed}
    for cfg in configs:
        try:
            it = iter(load_dataset(DATASET, cfg, split="test", streaming=True))
            for row in itertools.islice(it, args.test_rows_per_config):
                for r in convert(bekko_row, row, source=cfg):
                    test_bekko.append(slim(r)); held.add(state_key(r))
        except Exception as e:  # noqa: BLE001  subset has no test split
            print(f"[no test] {cfg}: {type(e).__name__}", flush=True)
    test_bekko = test_bekko[: args.max_test]
    print(f"held-out: typed test {len(test_typed)}, bekko test {len(test_bekko)}", flush=True)

    # ---- stream the training pool round-robin ----------------------------------------------
    streams = {}
    for cfg in configs:
        try:
            streams[cfg] = iter(load_dataset(DATASET, cfg, split="train", streaming=True))
        except Exception as e:  # noqa: BLE001
            print(f"[skip] {cfg}: {type(e).__name__}: {str(e)[:80]}", flush=True)
    per_cfg, n_train, n_val, dropped, t0 = Counter(), 0, [], 0, time.time()
    ftrain = open(out / "train.jsonl", "w", encoding="utf-8")
    while streams and n_train < args.n_train:
        for cfg in list(streams):
            rows = list(itertools.islice(streams[cfg], args.chunk))
            if len(rows) < args.chunk:
                del streams[cfg]
            for row in rows:
                for r in convert(bekko_row, row, source=cfg):
                    key = state_key(r)
                    if key in held:
                        dropped += 1
                        continue
                    if int(key[:8], 16) % args.val_mod == 0 and len(n_val) < args.max_val:  # ~0.1% of states -> val
                        n_val.append(slim(r)); held.add(key); continue
                    if int(key[:8], 16) % args.val_mod == 0:
                        continue  # val is full: keep these states out of train anyway
                    ftrain.write(json.dumps(slim(r), ensure_ascii=False) + "\n")
                    n_train += 1; per_cfg[cfg] += 1
            if n_train >= args.n_train:
                break
        print(f"  {n_train:>9,} decisions  {len(streams)} subsets still streaming  {time.time()-t0:.0f}s", flush=True)
    ftrain.close()

    def write(path, rows):
        with open(path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(slim(r), ensure_ascii=False) + "\n")
    write(out / "val.jsonl", n_val); write(out / "test_bekko.jsonl", test_bekko); write(out / "test_typed.jsonl", test_typed)
    print(f"train={n_train:,} val={len(n_val)} test_bekko={len(test_bekko)} test_typed={len(test_typed)} "
          f"(dropped {dropped} leaking decisions) in {time.time()-t0:.0f}s")
    print("decisions per subset (top 10):", per_cfg.most_common(10))
    if ERRORS:
        print(f"WARNING: conversion errors: {dict(ERRORS.most_common(5))}")
    if n_train == 0:
        sys.exit("no training data produced: schema likely changed - see errors above")


if __name__ == "__main__":
    main()
