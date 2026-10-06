#!/usr/bin/env bash
# Small first fine-tune on a Mac (Apple silicon): BGE-M3 + LoRA head on public typed-decision data.
# Needs internet access to huggingface.co. Writes everything to run.log.   bash scripts/run_local_mac.sh
set -euo pipefail
cd "$(dirname "$0")/.."
exec > >(tee -a run.log) 2>&1
echo "== start $(date)"

PY=""
for c in python3.12 python3.11 python3.10 python3; do
  if command -v "$c" >/dev/null && "$c" -c 'import sys; sys.exit(sys.version_info < (3, 10))'; then PY="$c"; break; fi
done
[ -n "$PY" ] || { echo "ERROR: need Python >= 3.10 (brew install python@3.12)"; exit 1; }
echo "python: $($PY --version)  ram: $(( $(sysctl -n hw.memsize) / 1073741824 )) GB"

[ -d .venv ] || "$PY" -m venv .venv
source .venv/bin/activate
pip install -q --upgrade pip
pip install -q -e ".[data]"
export PYTORCH_ENABLE_MPS_FALLBACK=1 TOKENIZERS_PARALLELISM=false
python -c "import torch; print('torch', torch.__version__, '| mps available:', torch.backends.mps.is_available())"

BACKBONE="${BACKBONE:-BAAI/bge-m3}"
MAXLEN="${MAXLEN:-512}"; HEADLEN="${HEADLEN:-256}"
OUT=checkpoints/bge_m3_typed

echo "== 1/4 data"
python scripts/build_typed_decisions.py --out data/typed --max_train "${MAX_TRAIN:-1500}" --max_val 300 --max_test 500
python scripts/validate_jsonl.py data/typed/train.jsonl --tokenizer "$BACKBONE" --max_length "$MAXLEN" --head_length "$HEADLEN"

echo "== 2/4 train (downloads BGE-M3 ~2.3 GB on first run)"
python -m jev.train --backbone "$BACKBONE" --train data/typed/train.jsonl --val data/typed/val.jsonl --out "$OUT" \
  --max_length "$MAXLEN" --head_length "$HEADLEN" --lora 16 --grad_ckpt \
  --batch_size 4 --grad_accum 4 --epochs "${EPOCHS:-1}" --lr_head 3e-4 --lr_encoder 1e-4

echo "== 3/4 held-out test (LocalLLaMA/typed-decisions test split)"
python -m jev.evaluate --checkpoint "$OUT" --data data/typed/test.jsonl --max_length "$MAXLEN" --head_length "$HEADLEN" --batch_size 8 --show 10

echo "== 4/4 done $(date)"
ls -la "$OUT"
