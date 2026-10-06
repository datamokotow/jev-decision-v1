"""Inference API: one state, several named questions -> probabilities.

    decider = Decider("checkpoints/stage2", backbone="/models/bge-m3")
    out = decider.decide(state, {
        "route":    {"type": "choice", "instructions": "Which model should handle this?",
                     "criteria": {"haiku": "cheap, simple tasks", "opus": "hard multi-step reasoning"}},
        "valid":    {"type": "noul",   "instructions": "Is this review finding correct?"},
        "severity": {"type": "score",  "instructions": "How severe is it?",
                     "criteria": ["nit", "minor", "major", "critical"]},
    })
"""
import torch

from .data import Collator, validate_row
from .device import pick_device


class Decider:
    def __init__(self, checkpoint, backbone=None, device=None, max_length=2048, head_length=512, keep="head"):
        from transformers import AutoTokenizer
        from .model import DecisionModel, tokenizer_path
        self.device = device or pick_device()
        self.model = DecisionModel.from_pretrained(checkpoint, backbone=backbone).to(self.device).eval()
        self.collate = Collator(AutoTokenizer.from_pretrained(tokenizer_path(checkpoint, backbone)),
                                max_length, head_length, keep)

    @torch.inference_mode()
    def logits(self, rows):
        batch = {k: v.to(self.device) for k, v in self.collate(rows).items()}
        scores = self.model(**batch).float().cpu()
        return [s[: len(r["options"])].tolist() for s, r in zip(scores, rows)]

    def decide(self, state, questions):
        rows, meta = [], []
        for qid, q in questions.items():
            kind, crit = q["type"], q.get("criteria")
            if kind == "choice":
                keys, labels = list(crit), list(crit.values())
            elif kind == "score":
                labels, keys = list(crit), [str(i) for i in range(len(crit))]
            elif kind == "noul":
                keys, labels = ["false", "true"], list((crit or {"false": "false", "true": "true"}).values())
            else:
                raise ValueError(f"unknown type {kind}")
            row = dict(state=state, question=q["instructions"], type=kind, options=labels)
            validate_row(row, qid)
            rows.append(row); meta.append((qid, kind, keys))
        answers = {}
        for (qid, kind, keys), z in zip(meta, self.logits(rows)):
            p = torch.tensor(z).softmax(-1).tolist()
            res = dict(type=kind, probabilities=dict(zip(keys, p)))
            if kind == "choice":
                res["choice"] = keys[max(range(len(p)), key=p.__getitem__)]
            elif kind == "score":
                res["score"] = sum(i * x for i, x in enumerate(p))
            else:
                res["noul"] = p[1]
            if kind != "noul":
                res["max_probability"] = max(p)
            answers[qid] = res
        return answers
