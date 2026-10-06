"""End-to-end CPU smoke test with a tiny local encoder (no network, no HF hub)."""
import json
import random
from pathlib import Path

import pytest
import torch

from jev.data import Collator, Decisions, encode_row, validate_row
from jev.evaluate import evaluate
from jev.model import DecisionModel
from jev import train as train_mod

WORDS = [f"w{i}" for i in range(40)]


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast, RobertaConfig, RobertaModel
    root = tmp_path_factory.mktemp("tiny")
    specials = ["<pad>", "<s>", "</s>", "<unk>", "<mask>"]
    vocab = {t: i for i, t in enumerate(specials + WORDS + ["question", ":", "choice", "score", "noul"])}
    tk = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    tk.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tok = PreTrainedTokenizerFast(tokenizer_object=tk, pad_token="<pad>", cls_token="<s>", sep_token="</s>",
                                  unk_token="<unk>", mask_token="<mask>")
    tok.save_pretrained(root / "backbone")
    cfg = RobertaConfig(vocab_size=len(vocab), hidden_size=64, num_hidden_layers=2, num_attention_heads=2,
                        intermediate_size=128, max_position_embeddings=300, pad_token_id=0,
                        bos_token_id=1, eos_token_id=2)
    torch.manual_seed(0)
    RobertaModel(cfg).save_pretrained(root / "backbone")
    return root


def make_rows(n, seed):
    """Target = the option whose word also occurs in the state (needs real attention to solve)."""
    rnd = random.Random(seed)
    rows = []
    for _ in range(n):
        k = rnd.randint(2, 5)
        opts = rnd.sample(WORDS, k)
        t = rnd.randrange(k)
        noise = rnd.sample([w for w in WORDS if w not in opts], 6)
        state = " ".join(rnd.sample(noise + [opts[t]], 7))
        rows.append(dict(state=state, question="question : pick", options=opts, target=t, task="toy"))
    return rows


def make_easy(n, seed):
    """Target = option with the smallest word id. Learnable by a random-init tiny encoder in ~1.5k steps."""
    rnd = random.Random(seed)
    rows = []
    for _ in range(n):
        k = rnd.randint(2, 5)
        opts = rnd.sample(WORDS, k)
        t = min(range(k), key=lambda i: int(opts[i][1:]))
        rows.append(dict(state="w0", question="question : pick", options=opts, target=t, task="toy"))
    return rows


def write(path, rows):
    Path(path).write_text("\n".join(json.dumps(r) for r in rows))


def test_validate_rejects_bad_rows():
    good = dict(state="a", question="q", options=["x", "y"], target=1)
    validate_row(good)
    for bad in (dict(good, options=["x"]), dict(good, target=5), dict(good, type="noul", options=["a", "b", "c"]),
                dict(good, teacher_logits=[0.1]), dict(good, question="")):
        with pytest.raises(ValueError):
            validate_row(bad)


def test_sequence_layout_and_truncation(tiny):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(tiny / "backbone")
    row = dict(state=" ".join(WORDS) * 5, question="q", options=["w1", "w2", "w3"], target=0)
    for keep in ("head", "tail", "ends"):
        e = encode_row(tok, row, max_length=64, head_length=16, keep=keep)
        assert len(e["ids"]) <= 64 and e["truncated"]
        assert all(e["ids"][m] == tok.mask_token_id for m in e["markers"])
    # a mask token smuggled into the text must not create an extra marker
    e = encode_row(tok, dict(state="<mask> w1", question="q <mask>", options=["w1", "w2"], target=0), 64, 16)
    assert sum(i == tok.mask_token_id for i in e["ids"]) == 2


def _train_args(tiny, train, out, val=None, extra=()):
    args = ["--backbone", str(tiny / "backbone"), "--train", str(train), "--out", str(out),
            "--max_length", "64", "--head_length", "32", "--batch_size", "32", "--grad_accum", "1",
            "--workers", "0", "--head_layers", "1", "--lr_head", "1e-3", "--lr_encoder", "1e-3"]
    if val:
        args += ["--val", str(val)]
    return args + list(extra)


def _reload_matches(tiny, out, val, expect_acc):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(tiny / "backbone")
    col = Collator(tok, 64, 32)
    m = DecisionModel.from_pretrained(out, backbone=str(tiny / "backbone")).eval()
    assert abs(evaluate(m, Decisions(val), col)["accuracy"] - expect_acc) < 1e-6


def test_full_finetune_roundtrip(tiny, tmp_path):
    """Encoder is trained -> checkpoint must carry the encoder, and reloading reproduces eval exactly."""
    train, val, out = tmp_path / "train.jsonl", tmp_path / "val.jsonl", tmp_path / "ckpt"
    write(train, make_easy(3000, 1)); write(val, make_easy(200, 2))
    train_mod.main(_train_args(tiny, train, out, val, ["--epochs", "16"]))
    metrics = json.loads((out / "metrics.json").read_text())
    chance = sum(1 / len(r["options"]) for r in make_easy(200, 2)) / 200
    assert metrics["accuracy"] > chance + 0.2, (metrics["accuracy"], chance)
    assert json.loads((out / "jev_config.json").read_text())["encoder"] == "full"
    assert (out / "encoder").exists() and (out / "tokenizer").exists()
    _reload_matches(tiny, out, val, metrics["accuracy"])  # backbone arg is ignored for full checkpoints


def test_frozen_encoder_is_head_only_and_tiny(tiny, tmp_path):
    rows = make_rows(64, 3)
    for r in rows:
        r["teacher_logits"] = [5.0 if i == r["target"] else 0.0 for i in range(len(r["options"]))]
    train, val, out = tmp_path / "t.jsonl", tmp_path / "v.jsonl", tmp_path / "ckpt"
    write(train, rows); write(val, make_rows(32, 4))
    train_mod.main(_train_args(tiny, train, out, val, ["--epochs", "2", "--freeze_encoder"]))
    assert json.loads((out / "jev_config.json").read_text())["encoder"] == "none"
    assert not (out / "encoder").exists() and not (out / "adapter").exists()
    metrics = json.loads((out / "metrics.json").read_text())
    _reload_matches(tiny, out, val, metrics["accuracy"])
    with pytest.raises(ValueError):  # head-only checkpoint with no backbone anywhere
        cfg = json.loads((out / "jev_config.json").read_text()); cfg["backbone"] = None
        (out / "jev_config.json").write_text(json.dumps(cfg))
        DecisionModel.from_pretrained(out)


def test_lora_adapter_roundtrip(tiny, tmp_path):
    train, val, out = tmp_path / "t.jsonl", tmp_path / "v.jsonl", tmp_path / "ckpt"
    write(train, make_easy(300, 1)); write(val, make_easy(50, 2))
    train_mod.main(_train_args(tiny, train, out, val, ["--epochs", "2", "--lora", "4", "--grad_ckpt"]))
    assert json.loads((out / "jev_config.json").read_text())["encoder"] == "adapter"
    assert (out / "adapter").exists() and not (out / "encoder").exists()
    _reload_matches(tiny, out, val, json.loads((out / "metrics.json").read_text())["accuracy"])


def test_typed_decider(tiny, tmp_path):
    from jev.infer import Decider
    out = tmp_path / "ckpt"
    m = DecisionModel.from_backbone(str(tiny / "backbone"), head_layers=1)
    m.save_pretrained(out, backbone_name=str(tiny / "backbone"), mode="none")
    d = Decider(out, max_length=64, head_length=32)
    r = d.decide("w1 w2 w3", {
        "route": dict(type="choice", instructions="pick", criteria={"a": "w1", "b": "w2", "c": "w3"}),
        "ok": dict(type="noul", instructions="ok"),
        "sev": dict(type="score", instructions="sev", criteria=["w1", "w2", "w3", "w4"]),
    })
    assert set(r["route"]["probabilities"]) == {"a", "b", "c"}
    assert abs(sum(r["route"]["probabilities"].values()) - 1) < 1e-5
    assert 0 <= r["ok"]["noul"] <= 1 and 0 <= r["sev"]["score"] <= 3


def test_lora_with_grad_ckpt_actually_updates_adapter(tiny, tmp_path):
    """Guards the classic failure: LoRA + checkpointing + frozen embeddings -> adapter silently gets no grads."""
    import torch
    from jev.data import Collator
    from transformers import AutoTokenizer
    from peft import LoraConfig, get_peft_model
    enc = DecisionModel.from_backbone(str(tiny / "backbone"), head_layers=1)
    enc.encoder = get_peft_model(enc.encoder, LoraConfig(r=4, target_modules=train_mod.lora_regex("query,key,value,dense")))
    enc.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    enc.train()
    col = Collator(AutoTokenizer.from_pretrained(tiny / "backbone"), 64, 32)
    b = col(make_easy(8, 0))
    loss = torch.nn.functional.cross_entropy(enc(**{k: v for k, v in b.items() if k != "labels"}), b["labels"])
    loss.backward()
    lora = [p for n, p in enc.named_parameters() if "lora_B" in n]
    assert lora and all(p.grad is not None for p in lora)


def test_soft_target_training_learns(tiny, tmp_path):
    """Rows carry only target_probs (no hard target) - the trainer must learn from the distribution."""
    def soften(rows):
        out = []
        for r in rows:
            k = len(r["options"]); p = [0.1 / (k - 1)] * k; p[r["target"]] = 0.9
            out.append({x: y for x, y in r.items() if x != "target"} | {"target_probs": p})
        return out
    train, val, out = tmp_path / "t.jsonl", tmp_path / "v.jsonl", tmp_path / "ckpt"
    write(train, soften(make_easy(3000, 1))); write(val, soften(make_easy(200, 2)))
    train_mod.main(_train_args(tiny, train, out, val, ["--epochs", "16"]))
    m = json.loads((out / "metrics.json").read_text())
    chance = sum(1 / len(r["options"]) for r in make_easy(200, 2)) / 200
    assert m["accuracy"] > chance + 0.2, (m["accuracy"], chance)


def test_lazy_dataset_matches_eager(tiny, tmp_path):
    from jev.data import Decisions
    rows = make_easy(50, 5)
    write(tmp_path / "d.jsonl", rows)
    eager, lazy = Decisions(tmp_path / "d.jsonl"), Decisions(tmp_path / "d.jsonl", lazy_mb=0)
    assert len(lazy) == len(eager) == 50
    assert all(lazy[i] == eager[i] for i in (0, 17, 49)) and lazy.rows[3:6] == eager.rows[3:6]


def test_time_limit_stops_training_and_saves(tiny, tmp_path):
    train, val, out = tmp_path / "t.jsonl", tmp_path / "v.jsonl", tmp_path / "ckpt"
    write(train, make_easy(2000, 1)); write(val, make_easy(40, 2))
    train_mod.main(_train_args(tiny, train, out, val, ["--epochs", "1000", "--time_limit_min", "0.05",
                                                       "--checkpoint_every_min", "0.02"]))
    assert (out / "jev_config.json").exists() and (out / "latest" / "jev_config.json").exists()
