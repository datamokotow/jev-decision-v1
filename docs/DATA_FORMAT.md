# Corp data format

One JSON object per line. The model reads `state`, weighs the described `options` for a `question`,
and the `target` is the index of the right option. Everything else is optional.

```json
{"state": "...text or JSON...", "question": "...", "type": "choice|score|noul",
 "options": ["...", "..."], "target": 0, "task": "tag for per-task metrics",
 "teacher_logits": [1.2, -0.3]}
```

| type   | options                                   | output                                  | use for                              |
|--------|-------------------------------------------|-----------------------------------------|--------------------------------------|
| choice | 2-20 descriptions, any order              | softmax over options                    | router, tool selection               |
| score  | ordered rubric, worst -> best (2-20)      | probabilities + expected level          | severity, eval metric levels         |
| noul   | exactly `[false description, true description]` | P(true)                           | validate a finding, pass/fail checks |

Rules the validator enforces: 2-20 options, each a non-empty string; `noul` has exactly 2; `target` indexes
`options`; `teacher_logits` (if present) match the option count.

Layout the model sees: `[CLS] <type> question: <question> [SEP] [MASK] opt1 [MASK] opt2 ... [SEP] <state> [SEP]`.
Question and options come first, so a long `state` is truncated, never the options. Options are cut at 48 tokens.

## Your four use cases

All four fit the three primitives. Write one row per decision; one source record often yields several rows.

### 1. Evals (context + eval metric output)  ->  `score` and `noul`
`state` = the context plus the model output (and any metric numbers). One row per metric.
Put the rubric in the options, ordered worst to best.

```json
{"state": "TASK: summarise the incident...\nOUTPUT: ...\nMETRICS: rouge_l=0.31, citations=2/5", "question": "How faithful is the output to the task context?", "type": "score", "options": ["Fabricates facts", "Several unsupported claims", "Minor unsupported detail", "Fully grounded"], "target": 2, "task": "eval_faithfulness"}
{"state": "...", "question": "Does the output answer the question that was asked?", "type": "noul", "options": ["Does not answer the question", "Answers the question"], "target": 1, "task": "eval_answers"}
```

### 2. Tool-use decisions (from Claude Code traces)  ->  `choice` (+ `noul`)
`state` = task + the **recent** trace (tool calls and abridged results). Train with `--keep tail` so the
most recent steps survive truncation. Options = the tools available at that step, as `name: what it does`.
Label = the tool actually used next in a trajectory that **succeeded**. Do not train on failed runs' actions.

```json
{"state": {"task": "fix failing test test_parse", "trace": [{"tool": "Bash", "cmd": "pytest -x", "result": "1 failed: test_parse..."}, {"tool": "Read", "path": "parser.py"}]}, "question": "Which tool should be called next?", "type": "choice", "options": ["Read: read a file", "Edit: modify a file", "Bash: run a shell command", "Grep: search file contents"], "target": 1, "task": "tool_use_next"}
{"state": "...same trace...", "question": "Is this proposed call safe to run without asking the user?", "type": "noul", "options": ["Needs user confirmation", "Safe to run automatically"], "target": 1, "task": "tool_use_safe"}
```

Scrub secrets, tokens and customer data from traces before they enter the dataset.

### 3. Critique validator (PR-review second pass)  ->  `noul` + `score`
`state` = the rule text, the diff hunk, and the reviewer's finding with its claimed severity.
Labels come from human outcomes on past reviews: finding accepted vs dismissed, severity as finally agreed.

```json
{"state": "RULE: no raw SQL string concatenation\nDIFF:\n+ q = 'SELECT * FROM u WHERE id=' + uid\nFINDING: SQL injection via uid (claimed: critical)", "question": "Is this finding a true violation of the rule?", "type": "noul", "options": ["False positive", "True violation"], "target": 1, "task": "critique_valid"}
{"state": "...same...", "question": "What is the correct severity of this finding?", "type": "score", "options": ["nit", "minor", "major", "critical"], "target": 3, "task": "critique_severity"}
```

Include plenty of **false positives** (dismissed findings). If almost all labels are "true" the model learns to approve everything.

### 4. LLM router  ->  `choice`
`state` = the prompt (plus cheap features if you have them: length, needs tools, language, domain).
Options = your models with a stable description of cost and capability. Label = the **cheapest model that
passed your eval** for that prompt. Labelling with "the best model" teaches it to always pick the most expensive.

```json
{"state": "Convert this CSV to JSON...", "question": "Which model should handle this request?", "type": "choice", "options": ["small-fast: cheap, simple extraction and formatting", "mid: general coding and analysis", "large: hard multi-step reasoning, long context"], "target": 0, "task": "router"}
```

Keep option descriptions identical across rows (the model learns them) and shuffle the order per row.

## Quality checklist
- Run `python scripts/validate_jsonl.py corp.jsonl --tokenizer <local bge-m3 path>` first: it reports label balance,
  target-position balance and how many states exceed `--max_length`.
- **Split by group, not by row**: by PR / repo / session / prompt-template, so near-duplicates do not leak into validation.
- Start with ~500-2,000 rows per use case; add more where `by_task` accuracy lags.
- Balance labels. For `noul`, aim for 30-70% true.
- Set `task` on every row: evaluation reports accuracy per tag.
- Optional distillation: put a stronger model's per-option scores in `teacher_logits`; `--distill_alpha` blends it with the labels.
