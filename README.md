# jev-decision

A small decision model: give it a **state**, a **question** and **described options**; it returns a probability
for each option. One model, three question types:

| type   | what it decides                          | example                                      |
|--------|------------------------------------------|----------------------------------------------|
| choice | pick 1 of 2-20 described options         | LLM router, next tool call                   |
| score  | ordered rubric level (soft score)        | eval metric level, finding severity          |
| noul   | boolean, returns P(true)                 | "is this review finding correct?"            |

Design is inspired by the public description of SupersonicLabs/Julia-1 (marker-scoring head on an encoder).
This is a clean-room implementation: Julia-1's training pipeline and data are not public, and nothing here is
derived from them beyond the published architecture idea.

## How it works
Input: `[CLS] <type> question: <q> [SEP] [MASK] opt1 [MASK] opt2 ... [SEP] <state> [SEP]`.
The encoder's hidden state at each `[MASK]` marker goes through a small 2-layer transformer head
(which sees only `[CLS]` + markers, so it is cheap) and an MLP scorer; softmax over markers is the decision.
Options sit before the state, so long states are truncated and options never are.
Works with any HF encoder that has a mask token: **BGE-M3** (default target, 8k context), XLM-R, RoBERTa, ModernBERT.

## Workflow: train outside, fine-tune in corp

```
Colab (HF reachable)                         GitHub                         Corp (no HF)
build public data  ──┐                                                      clone repo
stage-1 train        ├─ checkpoints/stage1 ──►  repo + head (~100 MB) ──►   point --backbone at local BGE-M3
                     ┘                                                      stage-2 fine-tune on corp JSONL
```

1. **Colab**: open `notebooks/stage1_colab.ipynb`. It builds the public data, trains stage 1 and pushes the checkpoint.
2. **Corp**: `git clone`, `git lfs pull`, `pip install -e .`, then:
   ```bash
   python scripts/validate_jsonl.py corp_train.jsonl --tokenizer /models/bge-m3
   python -m jev.train --backbone /models/bge-m3 --init checkpoints/stage1 \
       --train corp_train.jsonl --val corp_val.jsonl --out checkpoints/corp \
       --max_length 4096 --keep tail --bf16 --grad_ckpt --lora 16 --batch_size 2 --grad_accum 16
   ```
3. **Use it**:
   ```python
   from jev.infer import Decider
   d = Decider("checkpoints/corp", backbone="/models/bge-m3", max_length=4096, keep="tail")
   d.decide(state, {"route": {"type": "choice", "instructions": "Which model should handle this?",
                              "criteria": {"small": "cheap, simple tasks", "large": "hard reasoning"}}})
   ```

Data format for your four use cases (evals, tool use, critique validator, router): **docs/DATA_FORMAT.md**.

## Checkpoints: what gets saved
The save mode follows what was trained, so a checkpoint can never silently lose encoder changes:

| training                | saved                          | size (1024-wide backbone)   | loads with                       |
|-------------------------|--------------------------------|-----------------------------|----------------------------------|
| `--freeze_encoder`      | head only                      | ~100 MiB (50 MiB bf16)      | `backbone=` local encoder        |
| `--lora R`              | head + LoRA adapter            | ~100 MiB + a few MiB        | `backbone=` local encoder        |
| full fine-tune          | head + whole encoder           | ~2.3 GB                     | itself (too big for plain git)   |

Use `--lora` for anything that must travel through GitHub. The tokenizer is saved alongside (via Git LFS).
Git LFS and Releases both cap a single file at 2 GB, so a full BGE-M3 checkpoint belongs in a Release only if needed.

## Honest expectations
- Public data (stage 1) teaches the *format* and general reading of options. It will not know your rules, tools or
  models; stage-2 accuracy comes from your corp data.
- A frozen encoder is the weakest option: marker states of a retrieval-tuned encoder carry limited option detail. Prefer LoRA.
- Not a generator and not a reasoner: it picks among options you supply, grounded in the state. Check `by_task`
  accuracy on held-out groups (PRs, sessions) before trusting it, and use `max_probability` to defer low-confidence cases.
- The converters in `scripts/build_public_data.py` could not be run against the real HF datasets when this repo was written
  (HF was unreachable); they are unit-tested on fake rows. Dataset IDs/schemas can drift; sources that fail to load are
  skipped with a warning.
- The CPU test-suite trains a tiny encoder; no BGE-M3 training has been run. Expect to tune LR/batch on a real GPU.

## Layout
```
jev/model.py      DecisionModel, checkpoint save/load (none | adapter | full)
jev/data.py       JSONL schema + validation, sequence builder, collator
jev/train.py      CE (+ optional teacher KL), LoRA/freeze, bf16, grad checkpointing
jev/evaluate.py   accuracy / loss / confidence overall and per task
jev/infer.py      Decider: typed multi-question API
scripts/          build_public_data.py, validate_jsonl.py
tests/            CPU smoke tests (tiny encoder), converter tests
```
Tests: `pip install -e .[dev,data] && pytest -q` (about 1 minute on CPU).
