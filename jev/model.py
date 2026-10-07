"""Marker-scoring decision model.

Every candidate answer is rendered as ``[MASK] <description>`` in the input. The encoder
hidden state at each [MASK] marker is refined by a small transformer head (which only sees
[CLS] + the markers, so it is cheap) and scored by an MLP. A softmax over the markers gives
the decision. One model serves three question types:

    choice : pick one of N described options
    score  : ordered rubric, N levels (expected index = soft score)
    noul   : boolean, options are always [false, true]

The backbone is any HF encoder with a mask token (BGE-M3, XLM-R, RoBERTa, ModernBERT...).
Checkpoints can be saved head-only (small, GitHub friendly) and re-attached to a backbone
that already lives on the target machine.
"""
import json
from pathlib import Path

import torch
from torch import nn
from safetensors.torch import load_file, save_file

QTYPES = {"choice": 0, "score": 1, "noul": 2}
FORMAT_VERSION = 1
HEAD_FILE = "head.safetensors"
FULL_FILE = "model.safetensors"
CONFIG_FILE = "jev_config.json"


class DecisionModel(nn.Module):
    def __init__(self, encoder, head_layers=2, dropout=0.1):
        super().__init__()
        self.encoder = encoder
        width = encoder.config.hidden_size
        self.settings = dict(head_layers=head_layers, dropout=dropout)
        self.type_emb = nn.Embedding(len(QTYPES), width)
        nn.init.normal_(self.type_emb.weight, std=0.02)
        if head_layers:
            layer = nn.TransformerEncoderLayer(
                width, max(1, width // 64), 4 * width, dropout, activation="gelu",
                batch_first=True, norm_first=True)
            self.head = nn.TransformerEncoder(layer, head_layers, enable_nested_tensor=False)
        else:
            self.head = None
        self.scorer = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width), nn.GELU(),
                                    nn.Linear(width, 1))

    # ---- construction -------------------------------------------------------------
    @classmethod
    def from_backbone(cls, path, head_layers=2, dropout=0.1, **hf_kwargs):
        from transformers import AutoModel
        return cls(AutoModel.from_pretrained(path, **hf_kwargs), head_layers=head_layers, dropout=dropout)

    def head_parameters(self):
        return [p for n, p in self.named_parameters() if not n.startswith("encoder.")]

    # ---- forward ------------------------------------------------------------------
    def forward(self, input_ids, attention_mask, marker_pos, marker_mask, qtype, **_):
        hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        hidden = hidden + self.type_emb(qtype)[:, None, :].to(hidden.dtype)
        # position 0 ([CLS]) + markers: the head only ever sees this short sequence.
        pos = torch.cat((torch.zeros_like(marker_pos[:, :1]), marker_pos), dim=1)
        sel = hidden.gather(1, pos[:, :, None].expand(-1, -1, hidden.shape[-1]))
        if self.head is not None:
            keep = torch.cat((torch.ones_like(marker_mask[:, :1]), marker_mask), dim=1)
            sel = self.head(sel, src_key_padding_mask=~keep)
        scores = self.scorer(sel[:, 1:]).squeeze(-1).float()
        return scores.masked_fill(~marker_mask, -1e4)

    # ---- serialization ------------------------------------------------------------
    def encoder_mode(self):
        """What must be saved to reproduce this model: 'none' (encoder untouched, head-only checkpoint),
        'adapter' (LoRA deltas only) or 'full' (whole fine-tuned encoder)."""
        if hasattr(self.encoder, "peft_config"):
            return "adapter"
        return "full" if any(p.requires_grad for p in self.encoder.parameters()) else "none"

    def save_pretrained(self, directory, backbone_name=None, tokenizer=None, mode=None):
        """Writes jev_config.json + head.safetensors, plus encoder/ or adapter/ depending on `mode`
        (default: inferred from what is trainable). Head-only is ~100 MB for a 1024-wide backbone."""
        root = Path(directory)
        root.mkdir(parents=True, exist_ok=True)
        mode = mode or self.encoder_mode()
        config = dict(format_version=FORMAT_VERSION, architecture="DecisionModel",
                      backbone=backbone_name, encoder=mode, **self.settings)
        (root / CONFIG_FILE).write_text(json.dumps(config, indent=2) + "\n")
        save_file({k: v.detach().cpu().contiguous() for k, v in self.state_dict().items()
                   if not k.startswith("encoder.")}, str(root / HEAD_FILE), metadata={"format": "pt"})
        if mode == "adapter":
            self.encoder.save_pretrained(root / "adapter")
        elif mode == "full":
            enc = self.encoder.merge_and_unload() if hasattr(self.encoder, "peft_config") else self.encoder
            enc.save_pretrained(root / "encoder")
        if tokenizer is not None:
            tokenizer.save_pretrained(root / "tokenizer")

    @classmethod
    def from_pretrained(cls, directory, backbone=None, map_location="cpu", **hf_kwargs):
        """`backbone` = local path/id of the base encoder. Needed for head-only and adapter checkpoints
        (falls back to the name recorded at save time); ignored for full checkpoints, which carry their own."""
        from transformers import AutoModel
        root = Path(directory)
        config = json.loads((root / CONFIG_FILE).read_text())
        if config["format_version"] != FORMAT_VERSION:
            raise ValueError("Unsupported checkpoint format")
        mode = config["encoder"]
        if mode == "full":
            encoder = AutoModel.from_pretrained(str(root / "encoder"), **hf_kwargs)
        else:
            source = backbone or config.get("backbone")
            if not source:
                raise ValueError(f"'{mode}' checkpoint needs the base encoder: pass backbone=<path or id>")
            encoder = AutoModel.from_pretrained(str(source), **hf_kwargs)
            if mode == "adapter":
                from peft import PeftModel
                encoder = PeftModel.from_pretrained(encoder, str(root / "adapter"), is_trainable=True)
        model = cls(encoder, head_layers=config["head_layers"], dropout=config["dropout"])
        head_state = _read_head(root, map_location)
        missing, unexpected = model.load_state_dict(head_state, strict=False)
        bad = [k for k in missing if not k.startswith("encoder.")]
        if bad or unexpected:
            raise ValueError(f"Head mismatch. missing={bad[:5]} unexpected={unexpected[:5]}")
        return model


def _read_head(root, device="cpu"):
    """head.safetensors, or head.safetensors.part00, .part01, ... (checkpoint split to fit GitHub's 100 MB file limit)."""
    whole = root / HEAD_FILE
    if whole.exists():
        return load_file(str(whole), device=device)
    parts = sorted(root.glob(HEAD_FILE + ".part*"))
    if not parts:
        raise FileNotFoundError(f"no {HEAD_FILE} (or {HEAD_FILE}.part*) in {root}")
    from safetensors.torch import load
    state = load(b"".join(p.read_bytes() for p in parts))
    return {k: v.to(device) for k, v in state.items()}


def tokenizer_path(directory, backbone=None):
    """Tokenizer source for a checkpoint: explicit backbone > bundled tokenizer/ > recorded backbone name."""
    root = Path(directory)
    if backbone:
        return str(backbone)
    if (root / "tokenizer").exists():
        return str(root / "tokenizer")
    name = json.loads((root / CONFIG_FILE).read_text()).get("backbone")
    if not name:
        raise ValueError("No tokenizer found: pass backbone=<path or id>")
    return name
