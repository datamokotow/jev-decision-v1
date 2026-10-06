"""Accuracy / loss / calibration, overall and per `task` tag."""
import argparse
import json
from collections import defaultdict

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .device import pick_device


@torch.inference_mode()
def evaluate(model, dataset, collate, device="cpu", batch_size=8, amp=None):
    model.eval()
    ctx = amp if amp is not None else torch.autocast(device, enabled=False)
    loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate)
    n = correct = 0
    loss_sum = conf_sum = 0.0
    by = defaultdict(lambda: [0, 0])
    chance = defaultdict(float)
    idx = 0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with ctx:
            scores = model(**batch).float()
        loss_sum += -(batch["target_probs"] * F.log_softmax(scores, -1)).sum().item()
        prob = scores.softmax(-1)
        conf, pred = prob.max(-1)
        hit = (pred == batch["labels"]).tolist()
        for h, c in zip(hit, conf.tolist()):
            row = dataset[idx]; idx += 1
            tag = row.get("task") or row.get("type", "choice")
            by[tag][0] += h; by[tag][1] += 1
            chance[tag] += 1 / len(row["options"])
            correct += h; n += 1; conf_sum += c
    return dict(accuracy=correct / n, loss=loss_sum / n, mean_confidence=conf_sum / n, n=n,
                by_task={k: v[0] / v[1] for k, v in sorted(by.items())},
                chance_by_task={k: chance[k] / v[1] for k, v in sorted(by.items())})


@torch.inference_mode()
def show(model, dataset, collate, device, n=8):
    """Print n predictions next to the gold distribution (readable sanity check)."""
    model.eval()
    for row in dataset.rows[:n]:
        b = {k: v.to(device) for k, v in collate([row]).items()}
        p = model(**b).float().softmax(-1)[0, : len(row["options"])].tolist()
        gold = row.get("target_probs") or [float(i == row["target"]) for i in range(len(row["options"]))]
        top, gtop = max(range(len(p)), key=p.__getitem__), max(range(len(gold)), key=gold.__getitem__)
        flag = "OK " if top == gtop else "MISS"
        print(f"[{flag}] {row.get('task', row.get('type'))}: {row['question'][:70]}")
        print(f"       pred ({p[top]:.2f}): {row['options'][top][:80]}")
        print(f"       gold ({gold[gtop]:.2f}): {row['options'][gtop][:80]}")


def main(argv=None):
    from transformers import AutoTokenizer
    from .data import Collator, Decisions
    from .model import DecisionModel, tokenizer_path
    a = argparse.ArgumentParser()
    a.add_argument("--checkpoint", required=True)
    a.add_argument("--backbone", help="override/local backbone path")
    a.add_argument("--data", required=True)
    a.add_argument("--max_length", type=int, default=2048)
    a.add_argument("--head_length", type=int, default=512)
    a.add_argument("--keep", default="head")
    a.add_argument("--batch_size", type=int, default=8)
    a.add_argument("--summary", action="store_true", help="print overall numbers + best/worst 5 tasks only")
    a.add_argument("--fp16", action="store_true", help="fp16 autocast (CUDA); much faster at long context")
    a.add_argument("--show", type=int, default=0, help="also print N predictions vs gold")
    args = a.parse_args(argv)
    dev = pick_device()
    model = DecisionModel.from_pretrained(args.checkpoint, backbone=args.backbone).to(dev)
    tok = AutoTokenizer.from_pretrained(tokenizer_path(args.checkpoint, args.backbone))
    data = Decisions(args.data)
    col = Collator(tok, args.max_length, args.head_length, args.keep)
    amp = torch.autocast(dev, dtype=torch.float16, enabled=args.fp16 and dev == 'cuda')
    m = evaluate(model, data, col, dev, args.batch_size, amp)
    if args.summary:
        gap = {k: m["by_task"][k] - m["chance_by_task"][k] for k in m["by_task"]}
        up = sum(g > 0 for g in gap.values())
        print(json.dumps({k: v for k, v in m.items() if k not in ("by_task", "chance_by_task")}, indent=2))
        print(f"chance (mean over rows): {sum(m['chance_by_task'][k] for k in gap) / len(gap):.3f}  "
              f"tasks above chance: {up}/{len(gap)}")
        order = sorted(gap, key=gap.get)
        for title, ks in (("worst", order[:5]), ("best", order[-5:])):
            print(title, {k: f"{m['by_task'][k]:.2f} vs {m['chance_by_task'][k]:.2f}" for k in ks})
    else:
        print(json.dumps(m, indent=2))
    if args.show:
        show(model, data, col, dev, args.show)


if __name__ == "__main__":
    main()
