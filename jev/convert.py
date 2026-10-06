"""Converters from the public typed-decision datasets into decision JSONL (with soft targets).

Supported sources (schemas verified against rows from the HF dataset viewer):
  * LocalLLaMA/typed-decisions       : state / questions / gold (JSON strings)      -> typed_decisions_row()
  * hotchpotch/bekko-system-one-dataset-v0 : input.state_json + input.decisions[] + targets[] -> bekko_row()

One source row yields one decision row per question. Targets are the dataset's probability distributions.
"""
import json
import random

from .model import QTYPES

MAX_OPTIONS = 20


def _unjson(value):
    """Bekko stores strings JSON-encoded (e.g. '"Halt the agent now."'); tolerate plain strings."""
    if not isinstance(value, str):
        return "" if value is None else str(value)
    try:
        v = json.loads(value)
    except (ValueError, TypeError):
        return value
    return v if isinstance(v, str) else value


def _loads(value):
    if isinstance(value, str):
        try:
            return json.loads(value)
        except ValueError:
            return value
    return value


def _finish(state, question, kind, labels, probs, task, group):
    if kind not in QTYPES or not 2 <= len(labels) <= MAX_OPTIONS or len(labels) != len(probs):
        return None
    total = sum(probs)
    if total <= 0 or any(not (p >= 0) for p in probs):
        return None
    probs = [p / total for p in probs]
    labels = [l if l else str(i) for i, l in enumerate(labels)]
    if not question:
        return None
    return dict(state=state, question=question, type=kind, options=labels, target_probs=probs,
                target=max(range(len(probs)), key=probs.__getitem__), task=task, group=group)


def typed_decisions_row(row, rnd=None):
    """LocalLLaMA/typed-decisions row -> list of decision rows. Choice options are shuffled
    (the source lists them alphabetically, which would let position leak the label)."""
    rnd = rnd or random.Random(0)
    state = _loads(row["state"])
    questions, gold = _loads(row["questions"]), _loads(row["gold"])
    out = []
    for qid, q in questions.items():
        g = gold.get(qid)
        if not g:
            continue
        kind, crit = q.get("type"), q.get("criteria")
        if kind == "noul":  # criteria may be omitted: labels are then implicit [false, true]
            keys = ["false", "true"]
            labels = [crit.get(k) or k for k in keys] if isinstance(crit, dict) else keys
        elif kind == "score" and isinstance(crit, list):
            keys, labels = [str(i) for i in range(len(crit))], list(crit)
        elif kind == "choice" and isinstance(crit, dict) and crit:
            keys = list(crit)
            rnd.shuffle(keys)
            labels = [crit[k] for k in keys]
        else:
            continue  # unsupported / malformed question: skip, never crash the whole build
        probs = [float(g["probabilities"].get(k, 0.0)) for k in keys]
        r = _finish(state, q.get("instructions") or qid, kind, labels, probs, f"td:{row.get('workflow', 'x')}:{qid}", row["id"])
        if r:
            out.append(r)
    return out


def bekko_row(row, source="bekko"):
    """bekko-system-one-dataset-v0 row -> list of decision rows (noul forced to [false, true])."""
    inp = row["input"]
    state = _loads(inp["state_json"])
    targets = {t["decision_id"]: dict(zip(t["ids"], t["probabilities"])) for t in row["targets"]}
    out = []
    for d in inp["decisions"]:
        kind, crit = d.get("type"), d.get("criteria") or []
        dist = targets.get(d["id"])
        if dist is None or kind not in QTYPES:
            continue
        if kind == "noul":
            by_id = {c["id"]: c for c in crit}
            if set(by_id) != {"false", "true"}:
                continue
            crit = [by_id["false"], by_id["true"]]
        ids = [c["id"] for c in crit]
        labels = [_unjson(c.get("description_json")) for c in crit]
        question = _unjson(d.get("instructions_json"))
        if d.get("system_prompt"):
            question = f"{d['system_prompt']}\n{question}"
        s = {"state": state, "documents": d["documents"]} if d.get("documents") else state
        r = _finish(s, question, kind, labels, [float(dist.get(i, 0.0)) for i in ids],
                    f"{source}:{d['id']}", row.get("group_id") or row.get("case_id"))
        if r:
            out.append(r)
    return out


def state_key(row):
    import hashlib
    return hashlib.sha256(json.dumps(row["state"], sort_keys=True).encode()).hexdigest()


def holdout_by_state(rows, frac, rnd):
    """Split rows into (rest, held) so that all decisions sharing a state land on the same side."""
    keys = sorted({state_key(r) for r in rows})
    rnd.shuffle(keys)
    held_keys = set(keys[: max(1, int(len(keys) * frac))]) if keys else set()
    held = [r for r in rows if state_key(r) in held_keys]
    rest = [r for r in rows if state_key(r) not in held_keys]
    return rest, held
