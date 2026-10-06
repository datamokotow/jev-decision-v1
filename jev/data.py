"""Decision JSONL schema, sequence builder and collator.

Row schema (one JSON object per line)::

    {
      "state":    str | dict | list,   # context: code diff, trace, eval transcript, prompt...
      "question": str,                 # what to decide
      "type":     "choice" | "score" | "noul",   # default "choice"
      "options":  [str, ...],          # 2-20 non-empty descriptions (noul: [false_text, true_text])
      "target":   int,                 # index into options (hard label)
      "target_probs": [float, ...]     # OR a soft label: probabilities over options (annotator votes / teacher).
                                       #   if both are given, `target` is the argmax used for accuracy
      "task":     str,                 # optional tag, used for per-task metrics
      "teacher_logits": [float, ...]   # optional, same length as options, for distillation
    }

Sequence layout:  [CLS] "<type> question: <q>" [SEP] [MASK] opt1 [MASK] opt2 ... [SEP] <state> [SEP]
Question and options go FIRST so they are never truncated away when the state is long.
"""
import json
import math
from pathlib import Path

import torch
from torch.utils.data import Dataset

from .model import QTYPES

MAX_OPTION_TOKENS = 48


def validate_row(row, where="row"):
    p = f"{where}: "
    if not isinstance(row, dict):
        raise ValueError(p + "must be a JSON object")
    if not isinstance(row.get("state"), (str, dict, list)):
        raise ValueError(p + "state must be str/dict/list")
    if not isinstance(row.get("question"), str) or not row["question"]:
        raise ValueError(p + "question must be non-empty text")
    options = row.get("options")
    if not isinstance(options, list) or not 2 <= len(options) <= 20 \
            or not all(isinstance(x, str) and x for x in options):
        raise ValueError(p + "options must be 2-20 non-empty strings")
    kind = row.get("type", "choice")
    if kind not in QTYPES:
        raise ValueError(p + "type must be choice, score or noul")
    if kind == "noul" and len(options) != 2:
        raise ValueError(p + "noul needs exactly 2 options ordered [false, true]")
    if "target" in row and (type(row["target"]) is not int or not 0 <= row["target"] < len(options)):
        raise ValueError(p + "target must index options")
    tp = row.get("target_probs")
    if tp is not None:
        if (not isinstance(tp, list) or len(tp) != len(options)
                or not all(type(x) in (int, float) and math.isfinite(x) and x >= 0 for x in tp)
                or sum(tp) <= 0):
            raise ValueError(p + "target_probs must be non-negative floats matching options, sum > 0")
    t = row.get("teacher_logits")
    if t is not None and (not isinstance(t, list) or len(t) != len(options)
                          or not all(type(x) in (int, float) and math.isfinite(x) for x in t)):
        raise ValueError(p + "teacher_logits must be finite floats matching options")


def read_jsonl(path, require_target=True):
    rows = []
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{n}: invalid JSON: {e.msg}") from e
            validate_row(row, f"{path}:{n}")
            if require_target and "target" not in row and "target_probs" not in row:
                raise ValueError(f"{path}:{n}: target or target_probs required")
            rows.append(row)
    if not rows:
        raise ValueError(f"{path}: no rows")
    return rows


class _LazyRows:
    """Random access into a large JSONL file without holding it in memory (byte-offset index)."""

    def __init__(self, path, require_target):
        self.path, self.require_target, self._fh = str(path), require_target, None
        offsets, pos = [], 0
        with open(path, "rb") as f:
            for line in f:
                if line.strip():
                    offsets.append(pos)
                pos += len(line)
        if not offsets:
            raise ValueError(f"{path}: no rows")
        self.offsets = offsets

    def __len__(self):
        return len(self.offsets)

    def _one(self, i):
        if self._fh is None:  # opened per process, so DataLoader workers each get their own handle
            self._fh = open(self.path, "rb")
        self._fh.seek(self.offsets[i])
        row = json.loads(self._fh.readline())
        validate_row(row, f"{self.path}[{i}]")
        if self.require_target and "target" not in row and "target_probs" not in row:
            raise ValueError(f"{self.path}[{i}]: target or target_probs required")
        return row

    def __getitem__(self, i):
        if isinstance(i, slice):
            return [self._one(j) for j in range(*i.indices(len(self)))]
        return self._one(i)

    def __getstate__(self):
        return {**self.__dict__, "_fh": None}


class Decisions(Dataset):
    """JSONL decisions. Files over `lazy_mb` MB are indexed and read on demand (millions of rows)."""

    def __init__(self, path, require_target=True, lazy_mb=200):
        big = Path(path).stat().st_size > lazy_mb * 1_000_000
        self.rows = _LazyRows(path, require_target) if big else read_jsonl(path, require_target)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        return self.rows[i]


def _clean(tok, text):
    return text.replace(tok.mask_token, " ")


def encode_row(tok, row, max_length=8192, head_length=512, keep="head"):
    """Tokenise one row. `keep` picks which part of an over-long state survives:
    'head' (start), 'tail' (end - best for traces/logs) or 'ends' (both)."""
    if head_length + 4 >= max_length:
        raise ValueError("max_length must exceed head_length + 4")
    enc = lambda t: tok(t, add_special_tokens=False)["input_ids"]
    kind = row.get("type", "choice")
    state = row["state"] if isinstance(row["state"], str) else json.dumps(row["state"], ensure_ascii=False)
    head = enc(f"{kind} question: {_clean(tok, row['question'])}")
    opts = [[tok.mask_token_id] + enc(" " + _clean(tok, o))[:MAX_OPTION_TOKENS] for o in row["options"]]
    budget = head_length - sum(map(len, opts))
    if budget < 16:  # many/long options: shrink each so the question still fits
        per = max(4, (head_length - 16) // len(opts))
        opts = [o[:per] for o in opts]
        budget = head_length - sum(map(len, opts))
    ids = [tok.cls_token_id] + head[:max(8, budget)] + [tok.sep_token_id]
    markers = []
    for o in opts:
        markers.append(len(ids))
        ids.extend(o)
    ids.append(tok.sep_token_id)
    room = max_length - len(ids) - 1
    if room < 1:
        raise ValueError("question+options exceed max_length")
    s = enc(_clean(tok, state))
    truncated = len(s) > room
    if truncated:
        if keep == "tail":
            s = s[-room:]
        elif keep == "ends":
            s = s[: room // 2] + s[-(room - room // 2):]
        else:
            s = s[:room]
    return dict(ids=ids + s + [tok.sep_token_id], markers=markers, qtype=QTYPES[kind], truncated=truncated)


class Collator:
    def __init__(self, tokenizer, max_length=8192, head_length=512, keep="head"):
        self.tok, self.max_length, self.head_length, self.keep = tokenizer, max_length, head_length, keep

    def __call__(self, rows):
        enc = [encode_row(self.tok, r, self.max_length, self.head_length, self.keep) for r in rows]
        length = min(self.max_length, ((max(len(e["ids"]) for e in enc) + 7) // 8) * 8)
        k = max(len(e["markers"]) for e in enc)
        ids = torch.full((len(rows), length), self.tok.pad_token_id, dtype=torch.long)
        attn = torch.zeros_like(ids)
        pos = torch.zeros((len(rows), k), dtype=torch.long)
        mask = torch.zeros_like(pos, dtype=torch.bool)
        teacher = torch.zeros_like(pos, dtype=torch.float32) if all("teacher_logits" in r for r in rows) else None
        for i, (r, e) in enumerate(zip(rows, enc)):
            n, m = len(e["ids"]), len(e["markers"])
            ids[i, :n] = torch.tensor(e["ids"])
            attn[i, :n] = 1
            pos[i, :m] = torch.tensor(e["markers"])
            mask[i, :m] = True
            if teacher is not None:
                teacher[i, :m] = torch.tensor(r["teacher_logits"], dtype=torch.float32)
        batch = dict(input_ids=ids, attention_mask=attn, marker_pos=pos, marker_mask=mask,
                     qtype=torch.tensor([e["qtype"] for e in enc]))
        if all("target" in r or "target_probs" in r for r in rows):
            soft = torch.zeros_like(pos, dtype=torch.float32)
            labels = []
            for i, r in enumerate(rows):
                m = len(r["options"])
                if "target_probs" in r:
                    tp = torch.tensor(r["target_probs"], dtype=torch.float32)
                    soft[i, :m] = tp / tp.sum()
                    labels.append(int(tp.argmax()) if "target" not in r else r["target"])
                else:
                    soft[i, r["target"]] = 1.0
                    labels.append(r["target"])
            batch["labels"] = torch.tensor(labels)
            batch["target_probs"] = soft
        if teacher is not None:
            batch["teacher_logits"] = teacher
        return batch
