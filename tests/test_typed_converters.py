"""Converters tested on real rows copied from the HF dataset viewer (tests/fixtures)."""
import json
import random
from pathlib import Path

import torch

from jev.convert import bekko_row, typed_decisions_row
from jev.data import validate_row
from jev.train import soft_ce

FIX = Path(__file__).parent / "fixtures"


def test_typed_decisions_row():
    rows = typed_decisions_row(json.loads((FIX / "typed_decisions_row.json").read_text()), random.Random(0))
    assert [r["type"] for r in rows] == ["choice", "noul", "score"]
    for r in rows:
        validate_row(r)
        assert abs(sum(r["target_probs"]) - 1) < 1e-6
    action, noul, risk = rows
    assert action["options"][action["target"]] == "Queue this trace for a human to review."
    assert noul["options"] == ["No human attention is warranted.", "A human should inspect this run."] and noul["target"] == 1
    assert risk["target"] == 0 and risk["target_probs"][1] > 0.4


def test_choice_order_is_shuffled_across_seeds():
    row = json.loads((FIX / "typed_decisions_row.json").read_text())
    orders = {tuple(typed_decisions_row(row, random.Random(s))[0]["options"]) for s in range(10)}
    assert len(orders) > 3


def test_bekko_row_matches_typed_decisions_row():
    b = bekko_row(json.loads((FIX / "bekko_row.json").read_text()))
    assert [r["type"] for r in b] == ["choice", "noul", "score"]
    for r in b:
        validate_row(r)
    assert b[0]["options"][b[0]["target"]] == "Queue this trace for a human to review."  # json-decoded text
    assert b[1]["options"][0] == "No human attention is warranted."                    # noul order [false, true]
    assert b[1]["target_probs"] == [0.45, 0.55] and b[1]["target"] == 1
    assert isinstance(b[0]["state"], dict) and b[0]["state"]["task"].startswith("Delete personal data")


def test_bekko_noul_is_reordered_and_bad_rows_skipped():
    row = json.loads((FIX / "bekko_row.json").read_text())
    nl = row["input"]["decisions"][1]
    nl["criteria"] = nl["criteria"][::-1]          # [true, false] in the source
    row["targets"][1]["ids"], row["targets"][1]["probabilities"] = ["true", "false"], [0.7, 0.3]
    r = bekko_row(row)[1]
    assert r["options"][0] == "No human attention is warranted." and r["target_probs"] == [0.3, 0.7]
    row["targets"] = []                             # no targets -> nothing emitted, no crash
    assert bekko_row(row) == []


def test_soft_ce_matches_hard_ce_for_onehot_and_masks_padding():
    scores = torch.tensor([[2.0, 0.5, -1e4]])
    mask = torch.tensor([[1.0, 1.0, 0.0]])
    hard = torch.nn.functional.cross_entropy(scores, torch.tensor([0]))
    assert torch.allclose(soft_ce(scores, torch.tensor([[1.0, 0.0, 0.0]]), mask), hard)
    soft = soft_ce(scores, torch.tensor([[0.5, 0.5, 0.0]]), mask)
    assert soft > 0 and torch.isfinite(soft)


def test_noul_without_criteria_and_malformed_question_skipped():
    row = json.loads((FIX / "typed_decisions_row.json").read_text())
    qs = json.loads(row["questions"])
    del qs["needs_review"]["criteria"]                       # real data omits criteria for some noul questions
    qs["bad"] = {"type": "choice", "instructions": "no criteria at all"}
    row["questions"] = json.dumps(qs)
    g = json.loads(row["gold"]); g["bad"] = g["action"]; row["gold"] = json.dumps(g)
    rows = typed_decisions_row(row, random.Random(0))
    assert [r["type"] for r in rows] == ["choice", "noul", "score"]   # 'bad' skipped, nothing raised
    assert rows[1]["options"] == ["false", "true"] and rows[1]["target"] == 1


def test_holdout_by_state_never_splits_a_state():
    from jev.convert import holdout_by_state, state_key
    rows = [dict(state={"id": i // 5}, question="q", type="noul", options=["a", "b"], target=0) for i in range(100)]
    rest, held = holdout_by_state(rows, 0.1, random.Random(0))
    assert len(held) == 10  # 2 of 20 states, 5 decisions each
    assert {state_key(r) for r in rest}.isdisjoint({state_key(r) for r in held})
    assert len(rest) + len(held) == 100
