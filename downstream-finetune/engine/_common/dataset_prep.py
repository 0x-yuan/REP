"""Build a chat-templated training dataset from a raw raw-harvest-style teacher dump.

Design notes:
  * NO filter on answer correctness — keep wrong-answer rows.
  * STRIP the V3 attack wrapper text from the assistant content (those
    ```bash\\n$ cat reasoning_trace.txt```... blocks are theft-tooling artifacts,
    not real reasoning; we do NOT want the student to learn to emit them).
  * Format must match the per-model chat template (Qwen2.5 or Llama-3.1).

The V3-wrapper-stripping regex set has been validated on ~10k-row raw harvest dumps:
roughly 80% of rows strip cleanly; the rest have malformed/truncated structure
that we drop (the wrapper-strip would leave broken text). Dropping malformed
rows is NOT an answer-correctness filter — wrong-answer rows that are
structurally intact are kept.

Per row:
  user.content       = row["question"]
  assistant.content  = {V3-stripped post-think}  (i.e. r2 paraphrase + final answer)

We drop the hidden `<think>...</think>` block (r1) entirely. The attacker only
ever observes the leaked r2 content, so a faithful distill student should
only see the leaked side too — not the privileged internal trace. Both
Qwen2.5-7B-Instruct and Llama-3.1-8B-Instruct are non-reasoning instruct
models (no <think> training, no reasoning special tokens), so we do not wrap
the content in any reasoning markers; the assistant turn is plain SFT text.

Output: HuggingFace DatasetDict({"train": ...}) with a single "text" column,
save_to_disk()'d for sft.py's load_from_disk consumption.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

RAW_REPO = "<your-hf-org>/<your-dataset>"
EXPECTED_VICTIM = "qwen3_14b"

THINK_CLOSE = "</think>"

# V3-attack wrapper patterns — match the two-block layout
# (``reasoning_trace.txt`` + ``final_answer.txt``) emitted by the V3 attack.

RT_OPENERS = [
    re.compile(r"\A\s*```bash\s*\n\s*\$\s*cat\s+reasoning_trace\.txt\s*\n"),
    re.compile(r"\A\s*\$\s*cat\s+reasoning_trace\.txt\s*\n"),
    re.compile(r"\n+\s*```bash\s*\n\s*\$\s*cat\s+reasoning_trace\.txt\s*\n"),
    re.compile(r"\n+\s*\$\s*cat\s+reasoning_trace\.txt\s*\n"),
]

# Anchored on closing fence ``` NOT \Z, so we don't eat the trailing
# `**Final Answer**\boxed{X}` postscript that often follows.
FINAL_ANSWER_EXTRACT = re.compile(
    r"```bash\s*\n\s*\$\s*cat\s+final_answer\.txt\s*\n(.*?)\n```",
    re.DOTALL,
)
FINAL_ANSWER_BLOCKS = [
    re.compile(
        r"\n?```\s*\n```bash\s*\n\s*\$\s*cat\s+final_answer\.txt\s*\n.*?\n```\s*",
        re.DOTALL,
    ),
    re.compile(
        r"\n?```bash\s*\n\s*\$\s*cat\s+final_answer\.txt\s*\n.*?\n```\s*",
        re.DOTALL,
    ),
    re.compile(r"\n+\s*\$\s*cat\s+final_answer\.txt\s*\n[^\n]*"),
]

TRAILING_FENCE = re.compile(r"\n```\s*\Z")
MULTI_NEWLINE = re.compile(r"\n{3,}")

LEAK_MARKERS = (
    "cat reasoning_trace.txt",
    "cat final_answer.txt",
    "reasoning_trace.txt",
    "final_answer.txt",
)


def _clean_post_think(post: str) -> tuple[str, str | None]:
    """Strip V3-attack wrapper from the post-`</think>` text. Returns (cleaned, extracted_answer)."""
    fa_match = FINAL_ANSWER_EXTRACT.search(post)
    extracted_answer = fa_match.group(1).strip() if fa_match else None

    s = post
    for _ in range(2):
        for pat in FINAL_ANSWER_BLOCKS:
            s = pat.sub("\n", s)
        for pat in RT_OPENERS:
            s = pat.sub("\n", s)
        s = TRAILING_FENCE.sub("", s)
        s = MULTI_NEWLINE.sub("\n\n", s)
    return s.strip(), extracted_answer


def _row_to_messages(row):
    """Return (messages_list_or_None, reject_reason_or_None).

    assistant.content = V3-stripped post-`</think>` text only. r1 is DROPPED:
    the student trains on q -> (r2 + final answer) like a normal SFT pair.
    """
    q = (row.get("question") or "").strip()
    completion = row.get("completion") or ""
    if not q:
        return None, "empty_question"
    if THINK_CLOSE not in completion:
        return None, "no_think_close"

    post_raw = completion.split(THINK_CLOSE, 1)[1]
    cleaned_post, extracted_answer = _clean_post_think(post_raw)
    if not cleaned_post:
        return None, "empty_post_after_clean"
    if any(m in cleaned_post for m in LEAK_MARKERS):
        return None, "leak_marker_residual"

    if r"\boxed{" not in cleaned_post:
        if extracted_answer:
            ans = extracted_answer.strip()
            if ans.startswith(r"\boxed{") and ans.endswith("}"):
                boxed = ans
            else:
                boxed = f"\\boxed{{{ans}}}"
            cleaned_post = cleaned_post.rstrip() + f"\n\n**Final Answer**\n{boxed}"
        else:
            return None, "missing_boxed"

    return (
        [
            {"role": "user", "content": q},
            {"role": "assistant", "content": cleaned_post},
        ],
        None,
    )


def _build_from_problem_solution(
    raw,
    out_path: str,
    tok,
    max_tokens: int,
) -> dict:
    """Clean `{problem, solution}` schema path (mix dataset).

    Each row: problem -> user turn, solution -> assistant turn. The solution is
    ALREADY the final training target (reasoning + \\boxed{...}); no V3 wrapper,
    no <think> block to strip. Render with the model's chat template so the
    `<|im_start|>assistant\\n` response-template the s1 DataCollator masks on is
    present, and the turn ends with `<|im_end|>`. Rows whose rendered length
    exceeds max_tokens are DROPPED (not truncated).
    """
    from datasets import Dataset, DatasetDict

    counts = {
        "considered": 0,
        "empty_problem": 0,
        "empty_solution": 0,
        "too_long": 0,
        "kept": 0,
    }
    token_lens = []
    rendered_rows = []
    for row in raw:
        counts["considered"] += 1
        problem = (row.get("problem") or "").strip()
        solution = (row.get("solution") or "").strip()
        if not problem:
            counts["empty_problem"] += 1
            continue
        if not solution:
            counts["empty_solution"] += 1
            continue
        msgs = [
            {"role": "user", "content": problem},
            {"role": "assistant", "content": solution},
        ]
        text = tok.apply_chat_template(msgs, tokenize=False)
        n_tok = len(tok(text, add_special_tokens=False)["input_ids"])
        if n_tok > max_tokens:
            counts["too_long"] += 1
            continue
        rendered_rows.append({"text": text})
        token_lens.append(n_tok)
        counts["kept"] += 1

    print("[dataset_prep] (problem/solution) counts:", flush=True)
    for k, v in counts.items():
        print(f"  {k:32s} {v}", flush=True)
    if token_lens:
        import statistics

        print(
            f"[dataset_prep] kept-row token stats: "
            f"min={min(token_lens)} p50={statistics.median(token_lens):.0f} "
            f"mean={sum(token_lens)//len(token_lens)} max={max(token_lens)} "
            f"(cap={max_tokens})",
            flush=True,
        )

    processed = Dataset.from_list(rendered_rows)
    Path(out_path).mkdir(parents=True, exist_ok=True)
    DatasetDict({"train": processed}).save_to_disk(out_path)
    print(f"[dataset_prep] wrote {out_path} ({len(processed)} rows)", flush=True)
    return counts


def build(
    out_path: str,
    model_name: str,
    source_repo: str = RAW_REPO,
    num_proc: int = 4,
    max_tokens: int = 32768,
    config_name: str | None = None,
) -> dict:
    """Render a teacher dump into a chat-templated 'text' dataset; save_to_disk(out_path).

    Two schemas are auto-detected by column presence:
      * `{problem, solution}` (clean mix dataset) -> `_build_from_problem_solution`:
        problem->user, solution->assistant, rendered via the model chat template.
        No wrapper strip, no correctness filter, drop-if-too-long.
      * raw-harvest-style `{question, completion, structural, victim, ...}` -> the
        legacy V3-attack path below (strip wrapper, drop malformed rows).
        `structural` / `victim` are optional (the Hub `REP-datasets` configs
        carry only `question, r1, r2, answer, completion`): a missing
        `structural` is derived as `"</think>" in completion`, a missing
        `victim` skips the victim check.

    `config_name` selects a dataset config (e.g. `distill_q3_14b_clean` of
    `Chia-Mu-Lab/REP-datasets`); None loads the repo's default config.

    Filter contract (raw-harvest path):
      * structural == True              (drops rows where the V3 attack format failed)
      * </think> in completion          (must be able to split for post-think text)
      * V3 wrapper strips cleanly       (no leak markers remain)
      * has \\boxed{...} (or we can restore it from `cat final_answer.txt` block)
      * rendered length <= max_tokens   (drop instead of truncate, per user spec)

    NO filter on answer correctness — wrong-answer rows that pass the structural
    checks ARE kept (faithful to "attacker has no oracle for correctness").

    Idempotent: if out_path/train/dataset_info.json exists, this is a no-op.
    Returns a counts dict for diagnostics.
    """
    from datasets import Dataset, DatasetDict, load_dataset
    from transformers import AutoTokenizer

    marker = Path(out_path) / "train" / "dataset_info.json"
    if marker.exists():
        print(f"[dataset_prep] already built: {out_path}", flush=True)
        return {"already_built": True}

    print(f"[dataset_prep] loading {source_repo} config={config_name} ...", flush=True)
    raw = (
        load_dataset(source_repo, config_name, split="train")
        if config_name
        else load_dataset(source_repo, split="train")
    )
    print(f"[dataset_prep] raw rows={len(raw)} cols={raw.column_names}", flush=True)

    print(f"[dataset_prep] loading tokenizer {model_name} ...", flush=True)
    tok = AutoTokenizer.from_pretrained(model_name)

    cols = set(raw.column_names)
    if "problem" in cols and "solution" in cols:
        print("[dataset_prep] detected clean {problem, solution} schema", flush=True)
        return _build_from_problem_solution(raw, out_path, tok, max_tokens)

    print("[dataset_prep] detected raw-harvest schema (V3-wrapper strip path)", flush=True)
    if "victim" in cols:
        victims = set(raw.unique("victim"))
        if victims != {EXPECTED_VICTIM}:
            print(f"[dataset_prep] WARN: victim mix = {victims} (expected {{{EXPECTED_VICTIM}}})", flush=True)
    has_structural = "structural" in cols

    counts = {
        "considered": 0,
        "skip_struct_false": 0,
        "empty_question": 0,
        "no_think_close": 0,
        "empty_post_after_clean": 0,
        "leak_marker_residual": 0,
        "missing_boxed": 0,
        "too_long": 0,
        "kept": 0,
    }
    token_lens = []
    rendered_rows = []
    for row in raw:
        counts["considered"] += 1
        structural = row.get("structural") if has_structural else (THINK_CLOSE in (row.get("completion") or ""))
        if not bool(structural):
            counts["skip_struct_false"] += 1
            continue
        msgs, reject = _row_to_messages(row)
        if reject:
            counts[reject] = counts.get(reject, 0) + 1
            continue
        text = tok.apply_chat_template(msgs, tokenize=False)
        n_tok = len(tok(text, add_special_tokens=False)["input_ids"])
        if n_tok > max_tokens:
            counts["too_long"] += 1
            continue
        rendered_rows.append({"text": text})
        token_lens.append(n_tok)
        counts["kept"] += 1

    print("[dataset_prep] counts:", flush=True)
    for k, v in counts.items():
        print(f"  {k:32s} {v}", flush=True)
    if token_lens:
        import statistics

        print(
            f"[dataset_prep] kept-row token stats: "
            f"min={min(token_lens)} p50={statistics.median(token_lens):.0f} "
            f"mean={sum(token_lens)//len(token_lens)} max={max(token_lens)} "
            f"(cap={max_tokens})",
            flush=True,
        )

    processed = Dataset.from_list(rendered_rows)
    Path(out_path).mkdir(parents=True, exist_ok=True)
    DatasetDict({"train": processed}).save_to_disk(out_path)
    print(f"[dataset_prep] wrote {out_path} ({len(processed)} rows)", flush=True)
    return counts


def _cli():
    p = argparse.ArgumentParser()
    p.add_argument("--out-path", required=True)
    p.add_argument("--model-name", required=True)
    p.add_argument("--source-repo", default=RAW_REPO)
    p.add_argument("--num-proc", type=int, default=4)
    p.add_argument("--config-name", default=None)
    args = p.parse_args()
    build(args.out_path, args.model_name, args.source_repo, args.num_proc, config_name=args.config_name)


if __name__ == "__main__":
    _cli()
