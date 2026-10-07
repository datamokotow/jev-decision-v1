# jev-bge-m3-v1

Marker-scoring decision model (choice / score / noul) on **BAAI/bge-m3** with a LoRA adapter (r=16) and a 2-layer decision head.
This folder holds only the adapter, head and tokenizer. **You need the original, unmodified BAAI/bge-m3 weights separately** (HF format: a folder with `config.json`, weights and tokenizer files).

The decision head is stored as `head.safetensors.part00/.part01` (GitHub's 100 MB file limit); `jev` joins them automatically when loading. To rebuild one file: `cat head.safetensors.part* > head.safetensors`.

## Training
- Data: Bekko System One dataset (`hotchpotch/bekko-system-one-dataset-v0`), 182 subsets streamed round-robin; 1,000,109 decisions in the pool.
- Kaggle T4, fp16, LoRA r=16, batch 4 x accum 8, head LR 3e-4, LoRA LR 2e-4, max_length 2048, head_length 512, `--keep ends`.
- 10 h wall clock, 5,135 optimizer steps, about 164k examples (16% of the pool), 4.56 samples/s.

## Results (held out)
| set | n | accuracy | chance |
|---|---|---|---|
| validation (same stream as train) | 438 | 58.9% | n/a |
| test_typed (LocalLLaMA/typed-decisions test) | 2000 | 48.7% | 31.7% |
| test_bekko (per-subset test splits) | 1541 | 55.7% | 33.1% |

Weak spots: agent-trace tasks (risk, urgency, needs_review) are at or below chance.

## Use / continue training
```
python -m jev.evaluate --checkpoint models/jev-bge-m3-v1 --backbone /path/to/bge-m3 --data your.jsonl --max_length 2048 --head_length 512 --keep ends --summary
python -m jev.train --backbone /path/to/bge-m3 --init models/jev-bge-m3-v1 --train corp.jsonl --out checkpoints/corp \
  --max_length 2048 --head_length 512 --keep tail --lora 16 --grad_ckpt --bf16 --epochs 3 --lr_head 1e-4 --lr_encoder 5e-5
```
`--init` on this checkpoint continues training the existing adapter (`--lora` is then ignored).
