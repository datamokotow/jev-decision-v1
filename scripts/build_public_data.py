"""Build stage-1 training data from public HF datasets (run where HF is reachable, e.g. Colab).

Each source is converted into the decision-JSONL format and mapped onto a Jev use case:

  router    <- banking77, massive            "which handler/route?"                 (choice)
  tool_use  <- xlam function-calling    "which tool should be called?"         (choice)
  critique  <- mnli                          "is the claim supported by evidence?"  (noul, + 3-way choice)
  evals     <- yelp_review_full              "how good is this? (5-level rubric)"   (score)
  general   <- ag_news, emotion              plain classification                   (choice)

Public data only teaches the *format* (read state, weigh described options, score the marker).
Your corp data (docs/DATA_FORMAT.md) is what specialises it. Sources that fail to load (gated,
renamed, offline) are skipped with a warning, so the script is safe to run as-is.

Licences: keep attribution (see DATA_LICENSES.md, written next to the output). Check each licence
before redistributing the converted JSONL.

    python scripts/build_public_data.py --out data --per_source 8000
"""
import argparse
import json
import random
from pathlib import Path

MAX_OPTIONS = 20

LICENSES = {
    "banking77": "PolyAI/banking77 - CC-BY-4.0",
    "massive": "AmazonScience/massive - CC-BY-4.0",
    "xlam": "Salesforce/xlam-function-calling-60k - CC-BY-4.0 (gated: accept terms on HF first)",
    "mnli": "nyu-mll/glue (MNLI) - mixed, see GLUE/MultiNLI terms (mostly OANC-derived; check before redistributing)",
    "yelp": "Yelp/yelp_review_full - Yelp dataset terms (non-commercial research); check before redistributing",
    "ag_news": "ag_news - academic, non-commercial use",
    "emotion": "dair-ai/emotion - educational/research use",
}


# ---------------------------------------------------------------- pure converters (unit tested)
def make_choice(state, question, labels, gold, rnd, task, max_options=MAX_OPTIONS, descriptions=None):
    """labels: all class names. Keeps gold + random distractors (cap max_options), shuffles order."""
    desc = descriptions or {}
    pool = [l for l in labels if l != gold]
    rnd.shuffle(pool)
    opts = [gold] + pool[: max_options - 1]
    rnd.shuffle(opts)
    return dict(state=state, question=question, type="choice", options=[desc.get(o, o) for o in opts],
                target=opts.index(gold), task=task)


def pretty(label):
    return label.replace("_", " ").replace(".", " ").strip()


def convert_classlabel(rows, names, text_key, question, task, rnd, descriptions=None):
    labels = [pretty(n) for n in names]
    return [make_choice(r[text_key], question, labels, labels[r["label"]], rnd, task, descriptions=descriptions)
            for r in rows if r.get("label", -1) >= 0]


def convert_mnli(rows, rnd):
    out = []
    for r in rows:
        if r["label"] < 0:
            continue
        state = f"Evidence: {r['premise']}\nClaim: {r['hypothesis']}"
        # critique-validator proxy: is the claim supported by the evidence?  (0 entailment)
        out.append(dict(state=state, question="Is the claim fully supported by the evidence?", type="noul",
                        options=["The claim is not supported by the evidence",
                                 "The claim is supported by the evidence"],
                        target=int(r["label"] == 0), task="critique_mnli_noul"))
        if rnd.random() < 0.5:  # three-way version teaches contradiction vs unrelated
            labels = ["supported by the evidence", "unrelated to the evidence", "contradicted by the evidence"]
            out.append(dict(state=state, question="How does the claim relate to the evidence?", type="choice",
                            options=labels, target=int(r["label"]), task="critique_mnli_choice"))
    return out


def convert_yelp(rows):
    rubric = ["Terrible: 1 star", "Poor: 2 stars", "Average: 3 stars", "Good: 4 stars", "Excellent: 5 stars"]
    return [dict(state=r["text"], question="Rate the overall quality described in the text.", type="score",
                 options=rubric, target=int(r["label"]), task="eval_score_yelp") for r in rows]


def convert_xlam(rows, rnd):
    out = []
    for r in rows:
        try:
            tools = json.loads(r["tools"]) if isinstance(r["tools"], str) else r["tools"]
            answers = json.loads(r["answers"]) if isinstance(r["answers"], str) else r["answers"]
            gold = answers[0]["name"]
        except Exception:
            continue
        table = {t["name"]: f"{t['name']}: {t.get('description', '')}".strip() for t in tools if "name" in t}
        if gold not in table or len(table) < 2:
            continue
        names = list(table)
        out.append(make_choice(r["query"], "Which tool should be called next?", names, gold, rnd,
                               "tool_use_xlam", descriptions=table))
    return out


# ---------------------------------------------------------------- dataset loading (needs `datasets`)
def sample(ds, n, rnd):
    idx = list(range(len(ds)))
    rnd.shuffle(idx)
    return [ds[i] for i in idx[:n]]


def build(per_source, seed, massive_locales):
    from datasets import load_dataset
    rnd = random.Random(seed)
    train, val = [], []

    def add(name, loader):
        try:
            tr, va = loader()
            train.extend(tr); val.extend(va)
            print(f"[ok]   {name}: {len(tr)} train / {len(va)} val")
        except Exception as e:  # gated, renamed, offline, schema drift -> skip, do not crash the run
            print(f"[skip] {name}: {type(e).__name__}: {str(e)[:120]}")

    def cls(ds_id, text_key, question, task, descriptions=None, config=None, tr="train", va="test"):
        def run():
            d = load_dataset(ds_id, config) if config else load_dataset(ds_id)
            names = d[tr].features["label"].names
            f = lambda split, n: convert_classlabel(sample(d[split], n, rnd), names, text_key, question, task, rnd, descriptions)
            return f(tr, per_source), f(va, max(200, per_source // 10))
        return run

    add("banking77", cls("PolyAI/banking77", "text", "Which banking support intent does the customer message match?", "router_banking77"))
    add("ag_news", cls("ag_news", "text", "Which news section does this article belong to?", "general_ag_news",
                       {"World": "world news", "Sports": "sports news", "Business": "business news", "Sci/Tech": "science and technology news"}))
    add("emotion", cls("dair-ai/emotion", "text", "Which emotion does the author express?", "general_emotion", va="validation"))

    def massive():
        tr_rows, va_rows = [], []
        for loc in massive_locales:
            d = load_dataset("AmazonScience/massive", loc)
            names = d["train"].features["intent"].names
            for split, bucket, n in (("train", tr_rows, per_source // len(massive_locales)),
                                     ("validation", va_rows, 100)):
                rows = [dict(utt=r["utt"], label=r["intent"]) for r in sample(d[split], n, rnd)]
                bucket.extend(convert_classlabel(rows, names, "utt", "Which handler should this assistant request be routed to?",
                                                 "router_massive", rnd))
        return tr_rows, va_rows
    add("massive", massive)

    def xlam():
        d = load_dataset("Salesforce/xlam-function-calling-60k")["train"]
        rows = convert_xlam(sample(d, per_source + 500, rnd), rnd)
        return rows[:per_source], rows[per_source:] or rows[:100]
    add("xlam", xlam)

    def mnli():
        d = load_dataset("nyu-mll/glue", "mnli")
        return (convert_mnli(sample(d["train"], per_source // 2, rnd), rnd),
                convert_mnli(sample(d["validation_matched"], 300, rnd), rnd))
    add("mnli", mnli)

    def yelp():
        d = load_dataset("Yelp/yelp_review_full")
        return (convert_yelp(sample(d["train"], per_source, rnd)), convert_yelp(sample(d["test"], 300, rnd)))
    add("yelp", yelp)

    rnd.shuffle(train)
    return train, val


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--out", default="data")
    a.add_argument("--per_source", type=int, default=8000)
    a.add_argument("--seed", type=int, default=0)
    a.add_argument("--massive_locales", nargs="+", default=["en-US", "de-DE", "es-ES", "fr-FR", "hi-IN"])
    args = a.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    train, val = build(args.per_source, args.seed, args.massive_locales)
    for name, rows in (("stage1_train.jsonl", train), ("stage1_val.jsonl", val)):
        with open(out / name, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    (out / "DATA_LICENSES.md").write_text("# Source licences\n\n" + "\n".join(f"- {v}" for v in LICENSES.values()) + "\n")
    print(f"wrote {len(train)} train / {len(val)} val rows to {out}")


if __name__ == "__main__":
    main()
