"""Train / fine-tune the decision model.

Stage 1 (Colab, public data):   python -m jev.train --backbone BAAI/bge-m3 --train data/stage1_train.jsonl ...
Stage 2 (corp, your data):      python -m jev.train --init checkpoints/stage1 --backbone /local/bge-m3 --train corp.jsonl ...
"""
import argparse
import json
import math
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .data import Collator, Decisions
from .model import DecisionModel
from .device import pick_device
from .evaluate import evaluate


def soft_ce(scores, target_probs, mask, label_smoothing=0.0):
    """Cross-entropy against a (possibly soft) distribution over the valid options."""
    if label_smoothing:
        k = mask.sum(-1, keepdim=True).clamp_min(1)
        target_probs = target_probs * (1 - label_smoothing) + label_smoothing * mask / k
    return -(target_probs * F.log_softmax(scores, -1)).sum(-1).mean()


def loss_fn(scores, batch, alpha, temperature, label_smoothing):
    ce = soft_ce(scores, batch["target_probs"], batch["marker_mask"].float(), label_smoothing)
    if "teacher_logits" not in batch or alpha <= 0:
        return ce
    t = temperature
    teacher = batch["teacher_logits"].masked_fill(~batch["marker_mask"], -1e4)
    kl = F.kl_div(F.log_softmax(scores / t, -1), F.softmax(teacher / t, -1), reduction="batchmean") * t * t
    return (1 - alpha) * ce + alpha * kl


def lora_regex(targets):
    """Regex over module names; excludes the (unused) pooler so no dead adapters are created."""
    return r"^(?!.*pooler).*\.(" + "|".join(t.strip() for t in targets.split(",")) + ")$"


def build_optimizer(model, lr_head, lr_encoder, wd):
    enc = [p for n, p in model.named_parameters() if n.startswith("encoder.") and p.requires_grad]
    head = [p for n, p in model.named_parameters() if not n.startswith("encoder.") and p.requires_grad]
    groups = [dict(params=head, lr=lr_head, weight_decay=wd)]
    if enc:
        groups.append(dict(params=enc, lr=lr_encoder, weight_decay=wd))
    return torch.optim.AdamW(groups)


def main(argv=None):
    a = argparse.ArgumentParser()
    a.add_argument("--backbone", required=True, help="HF id or local path of the encoder (e.g. BAAI/bge-m3)")
    a.add_argument("--init", help="checkpoint dir to continue from (stage-1 head)")
    a.add_argument("--train", required=True)
    a.add_argument("--val")
    a.add_argument("--out", required=True)
    a.add_argument("--max_length", type=int, default=2048)
    a.add_argument("--head_length", type=int, default=512)
    a.add_argument("--keep", default="head", choices=["head", "tail", "ends"], help="which part of long state survives")
    a.add_argument("--epochs", type=float, default=2)
    a.add_argument("--batch_size", type=int, default=4)
    a.add_argument("--grad_accum", type=int, default=8)
    a.add_argument("--lr_head", type=float, default=3e-4)
    a.add_argument("--lr_encoder", type=float, default=2e-5)
    a.add_argument("--weight_decay", type=float, default=0.01)
    a.add_argument("--warmup", type=float, default=0.05)
    a.add_argument("--label_smoothing", type=float, default=0.0)
    a.add_argument("--distill_alpha", type=float, default=0.5, help="used only when rows carry teacher_logits")
    a.add_argument("--distill_temp", type=float, default=2.0)
    a.add_argument("--freeze_encoder", action="store_true", help="train head only (GitHub-friendly checkpoint)")
    a.add_argument("--lora", type=int, default=0, help="LoRA rank on the encoder (needs peft); 0 = off")
    a.add_argument("--lora_targets", default="query,key,value,dense",
                   help="comma-separated module-name suffixes (ModernBERT: Wqkv,Wo,Wi,Wo)")
    a.add_argument("--head_layers", type=int, default=2)
    a.add_argument("--bf16", action="store_true")
    a.add_argument("--fp16", action="store_true", help="fp16 autocast + loss scaling (T4/P100 have no bf16)")
    a.add_argument("--time_limit_min", type=float, default=0,
                   help="wall-clock budget: LR schedule follows elapsed time and training stops at the limit")
    a.add_argument("--checkpoint_every_min", type=float, default=0,
                   help="every N minutes: evaluate (if --val) and save <out>/latest")
    a.add_argument("--grad_ckpt", action="store_true")
    a.add_argument("--max_steps", type=int, default=0)
    a.add_argument("--eval_every", type=int, default=0, help="steps; 0 = once per epoch")
    a.add_argument("--seed", type=int, default=0)
    a.add_argument("--workers", type=int, default=0)
    args = a.parse_args(argv)

    random.seed(args.seed); torch.manual_seed(args.seed)
    dev = pick_device()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.backbone)
    if args.init:
        model = DecisionModel.from_pretrained(args.init, backbone=args.backbone)
    else:
        model = DecisionModel.from_backbone(args.backbone, head_layers=args.head_layers)
    if args.freeze_encoder:
        for p in model.encoder.parameters():
            p.requires_grad = False
    elif args.lora and hasattr(model.encoder, "peft_config"):
        print("--init checkpoint already has a LoRA adapter: continuing to train it (ignoring --lora)")
    elif args.lora:
        from peft import LoraConfig, get_peft_model
        model.encoder = get_peft_model(model.encoder, LoraConfig(r=args.lora, lora_alpha=2 * args.lora,
                                       target_modules=lora_regex(args.lora_targets), lora_dropout=0.05))
    if args.grad_ckpt and not args.freeze_encoder:
        model.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.to(dev)

    train = Decisions(args.train)
    val = Decisions(args.val) if args.val else None
    collate = Collator(tok, args.max_length, args.head_length, args.keep)
    loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, collate_fn=collate,
                        num_workers=args.workers, drop_last=len(train) >= args.batch_size)
    total = args.max_steps or max(1, int(len(loader) * args.epochs / args.grad_accum))
    opt = build_optimizer(model, args.lr_head, args.lr_encoder, args.weight_decay)
    base = [g["lr"] for g in opt.param_groups]
    warm = max(1, int(total * args.warmup))
    use16 = args.fp16 and dev == "cuda"
    amp = torch.autocast(dev, dtype=torch.float16 if use16 else torch.bfloat16,
                         enabled=(args.bf16 or args.fp16) and dev == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use16)

    out = Path(args.out)
    best, step, micro, t0, running = -1.0, 0, 0, time.time(), []
    eval_every = args.eval_every or max(1, int(len(loader) / args.grad_accum))
    limit_s = args.time_limit_min * 60
    last_ckpt, seen = t0, 0

    def progress():
        """0..1; the larger of step progress and (if a time limit is set) elapsed-time progress."""
        p = step / total
        return max(p, (time.time() - t0) / limit_s) if limit_s else p

    last_eval_step = -1

    def run_eval():
        nonlocal best, last_eval_step
        last_eval_step = step
        if not val:
            return
        m = evaluate(model, val, collate, dev, args.batch_size, amp)
        print(f"[eval @ step {step}] acc={m['accuracy']:.4f} loss={m['loss']:.4f} "
              + " ".join(f"{k}={v:.3f}(chance {m['chance_by_task'][k]:.3f})" for k, v in m["by_task"].items()), flush=True)
        if m["accuracy"] > best:
            best = m["accuracy"]
            model.save_pretrained(out, backbone_name=args.backbone, tokenizer=tok)
            (out / "metrics.json").write_text(json.dumps(m, indent=2))
        model.train()

    model.train()
    while progress() < 1:
        for batch in loader:
            batch = {k: v.to(dev) for k, v in batch.items()}
            with amp:
                scores = model(**batch)
            loss = loss_fn(scores.float(), batch, args.distill_alpha, args.distill_temp, args.label_smoothing)
            scaler.scale(loss / args.grad_accum).backward()
            running.append(loss.item()); micro += 1; seen += len(batch["labels"]) if "labels" in batch else 0
            if micro % args.grad_accum:
                continue
            frac = progress()
            warm_frac = args.warmup
            scale = (frac + 1e-3) / warm_frac if frac < warm_frac else max(0.0, (1 - frac) / (1 - warm_frac))
            for g, b in zip(opt.param_groups, base):
                g["lr"] = b * scale
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_([p for g in opt.param_groups for p in g["params"]], 1.0)
            scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True); step += 1
            if step % 10 == 0:
                el = time.time() - t0
                print(f"step {step}/{total} loss={sum(running)/len(running):.4f} {el:.0f}s "
                      f"{seen/el:.2f} samples/s progress={progress():.1%}", flush=True)
                running.clear()
            if args.checkpoint_every_min and time.time() - last_ckpt > args.checkpoint_every_min * 60:
                run_eval()
                model.save_pretrained(out / "latest", backbone_name=args.backbone, tokenizer=tok)
                model.train(); last_ckpt = time.time()
            elif not args.checkpoint_every_min and (step % eval_every == 0 or step >= total):
                run_eval()
            if progress() >= 1:
                break
    if val and last_eval_step != step:
        run_eval()  # final evaluation; keeps the best-so-far checkpoint
    if not val:
        model.save_pretrained(out, backbone_name=args.backbone, tokenizer=tok)
    print("done. checkpoint:", out)


if __name__ == "__main__":
    main()
