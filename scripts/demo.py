"""Show what a trained checkpoint actually says on a few hand-written decisions.

    python scripts/demo.py checkpoints/quick --backbone BAAI/bge-small-en-v1.5
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from jev.infer import Decider  # noqa: E402

CASES = [
    ("router (banking intents)", "I was charged twice for the same coffee, can you reverse one?", {
        "route": dict(type="choice", instructions="Which banking support intent does the customer message match?",
                      criteria={"dup": "transaction charged twice", "lost": "lost or stolen card", "bal": "balance not updated",
                                "fx": "exchange rate question"})}),
    ("critique (claim vs evidence)", "Evidence: The function validates the email with a regex before saving.\nClaim: User input is never validated.", {
        "supported": dict(type="noul", instructions="Is the claim fully supported by the evidence?",
                          criteria={"false": "The claim is not supported by the evidence", "true": "The claim is supported by the evidence"})}),
    ("critique (supported)", "Evidence: The function validates the email with a regex before saving.\nClaim: Emails are checked before they are stored.", {
        "supported": dict(type="noul", instructions="Is the claim fully supported by the evidence?",
                          criteria={"false": "The claim is not supported by the evidence", "true": "The claim is supported by the evidence"})}),
    ("eval score (1-5 rubric)", "Waited 50 minutes, food arrived cold and the waiter was rude. Never again.", {
        "quality": dict(type="score", instructions="Rate the overall quality described in the text.",
                        criteria=["Terrible: 1 star", "Poor: 2 stars", "Average: 3 stars", "Good: 4 stars", "Excellent: 5 stars"])}),
    ("tool use (unseen domain!)", {"task": "fix failing test", "trace": [{"tool": "Bash", "cmd": "pytest -x", "result": "1 failed"}]}, {
        "next": dict(type="choice", instructions="Which tool should be called next?",
                     criteria={"Read": "Read: read a file", "Edit": "Edit: modify a file", "Bash": "Bash: run a shell command"})}),
]


def main():
    a = argparse.ArgumentParser()
    a.add_argument("checkpoint")
    a.add_argument("--backbone")
    a.add_argument("--max_length", type=int, default=320)
    a.add_argument("--head_length", type=int, default=160)
    args = a.parse_args()
    d = Decider(args.checkpoint, backbone=args.backbone, max_length=args.max_length, head_length=args.head_length)
    for title, state, qs in CASES:
        print(f"\n== {title}")
        for qid, r in d.decide(state, qs).items():
            probs = {k: round(v, 3) for k, v in r["probabilities"].items()}
            print(f"  {qid}: {json.dumps(probs)}", f"-> {r.get('choice') or ('P(true)=%.2f' % r['noul'] if 'noul' in r else 'score=%.2f' % r['score'])}")
    print("\nNote: the tool-use case is outside the public training data; expect it to be weak until stage 2.")


if __name__ == "__main__":
    main()
