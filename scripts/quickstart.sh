#!/usr/bin/env bash
# One-command first run on a laptop (needs internet access to huggingface.co, Python 3.10+).
# Small English encoder + ~6k public examples. Apple GPU (mps) / CUDA used automatically; CPU also works but is slow.
#   bash scripts/quickstart.sh
set -euo pipefail
cd "$(dirname "$0")/.."
BACKBONE="${BACKBONE:-BAAI/bge-small-en-v1.5}"   # 33M params; swap for BAAI/bge-m3 on a real GPU
PER_SOURCE="${PER_SOURCE:-1200}"
EPOCHS="${EPOCHS:-2}"

[ -d .venv ] || python3 -m venv .venv
source .venv/bin/activate
pip install -q -e ".[data]"

python scripts/build_public_data.py --out data/quick --per_source "$PER_SOURCE" --massive_locales en-US
python scripts/validate_jsonl.py data/quick/stage1_train.jsonl --tokenizer "$BACKBONE" --max_length 320

python -m jev.train --backbone "$BACKBONE" --train data/quick/stage1_train.jsonl --val data/quick/stage1_val.jsonl \
  --out checkpoints/quick --max_length 320 --head_length 160 --epochs "$EPOCHS" \
  --batch_size 16 --grad_accum 2 --lr_head 5e-4 --lr_encoder 5e-5 --head_layers 2

echo; echo "=== held-out accuracy vs chance (per task) ==="
python - <<'PY'
import json
m = json.load(open("checkpoints/quick/metrics.json"))
print(f"overall accuracy {m['accuracy']:.3f} on {m['n']} rows")
for k, v in m["by_task"].items():
    print(f"  {k:24s} acc {v:.3f}   chance {m['chance_by_task'][k]:.3f}")
PY
python scripts/demo.py checkpoints/quick --backbone "$BACKBONE"
