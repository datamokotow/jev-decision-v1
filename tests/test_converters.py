"""Converters tested on fake rows shaped like the real HF datasets (HF itself is unreachable in CI)."""
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import build_public_data as B  # noqa: E402
from jev.data import validate_row  # noqa: E402


def test_choice_caps_options_and_keeps_gold():
    rnd = random.Random(0)
    labels = [f"intent_{i}" for i in range(77)]
    for _ in range(50):
        row = B.make_choice("text", "q?", [B.pretty(l) for l in labels], B.pretty(labels[5]), rnd, "t")
        validate_row(row)
        assert len(row["options"]) == 20 and row["options"][row["target"]] == "intent 5"
        assert len(set(row["options"])) == 20


def test_target_position_is_shuffled():
    rnd = random.Random(1)
    pos = {B.make_choice("s", "q", list("abcdef"), "c", rnd, "t")["target"] for _ in range(100)}
    assert len(pos) > 3


def test_mnli_noul_and_choice():
    rnd = random.Random(0)
    rows = B.convert_mnli([dict(premise="p", hypothesis="h", label=l) for l in (0, 1, 2, -1)], rnd)
    for r in rows:
        validate_row(r)
    noul = [r for r in rows if r["type"] == "noul"]
    assert [r["target"] for r in noul] == [1, 0, 0]  # only entailment is "supported"; label -1 dropped


def test_yelp_score_and_xlam_tools():
    r = B.convert_yelp([dict(text="great", label=4)])[0]
    validate_row(r); assert r["type"] == "score" and r["target"] == 4 and len(r["options"]) == 5
    tools = [dict(name="get_weather", description="weather by city"), dict(name="search", description="web search")]
    rows = B.convert_xlam([dict(query="rain in Pune?", tools=json.dumps(tools),
                                answers=json.dumps([dict(name="get_weather", arguments={"city": "Pune"})]))], random.Random(0))
    validate_row(rows[0]); assert rows[0]["options"][rows[0]["target"]].startswith("get_weather")
    assert B.convert_xlam([dict(query="x", tools="not json", answers="[]")], random.Random(0)) == []


def test_corp_format_examples_are_valid():
    from jev.data import read_jsonl
    rows = read_jsonl(Path(__file__).resolve().parents[1] / "examples" / "corp_format_examples.jsonl")
    assert {r["type"] for r in rows} == {"choice", "score", "noul"}
