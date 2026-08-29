"""Shared machinery for the REP prompt builders.

Each experiment builder (config_sweep / cross_dataset / cross_victim) uses these
helpers to: load the vendored OpenThoughts-500 victim test set, render prompts
with the Qwen3 chat template under the deployed defender system prompt, and
write SGLang inbox JSONL files. The wrapper logic itself lives in
:mod:`rep_core.variants`.
"""
from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

# Make the rep_core package importable from any experiment folder.
_STEAL_METHOD = Path(__file__).resolve().parents[1]
if str(_STEAL_METHOD) not in sys.path:
    sys.path.insert(0, str(_STEAL_METHOD))

from rep_core.build_helpers import RENDER_MODEL_ID  # noqa: E402
from rep_core.prompt_primitives import build_defender_system  # noqa: E402

# The canonical victim test set (500 OpenThoughts-114k rows with the recorded
# benign internal trace r0 of one shadow model per config). Vendored as gzip
# under public/data/openthoughts_test_500/<config>.jsonl.gz.
DATA_ROOT = Path(__file__).resolve().parents[2] / "data"
OT_CONFIG = "ri_qwen3_14b"
OT_TEST_DIR = DATA_ROOT / "openthoughts_test_500"
OT_LOCAL = OT_TEST_DIR / f"{OT_CONFIG}.jsonl.gz"
SHOT_POOL_LOCAL = DATA_ROOT / "shot_pool" / "qwen3_14b.jsonl.gz"

SHOT_POOL_SIZE = 50
SHOT_POOL_SEED = 7


def load_jsonl_gz(p: Path) -> list[dict]:
    rows: list[dict] = []
    with gzip.open(p, "rt") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_ot500_test(config: str = OT_CONFIG) -> list[dict]:
    """500 OpenThoughts victim test rows. Schema per row:
    {idx, question, gold, meta, ri, answer, raw_generation, valid}.

    Reads the vendored gzip ``data/openthoughts_test_500/<config>.jsonl.gz``
    (configs: ri_qwen3_14b, ri_qwen3_32b).
    """
    path = OT_TEST_DIR / f"{config}.jsonl.gz"
    if not path.exists():
        raise FileNotFoundError(f"vendored test set missing: {path}")
    return load_jsonl_gz(path)


def load_ot_shot_pool(path: Path, size: int = SHOT_POOL_SIZE) -> list[dict]:
    """In-domain OpenThoughts shadow-demo pool as {q,r,a} triples.

    The vendored file (``data/shot_pool/*.jsonl.gz``) is the deterministic
    seed-7 50-row draw from the shadow model's demo pool (built with
    ``rep_core/prepare_shot_pool.py``). It is pre-sampled, so we slice the first
    ``size`` rows for byte-determinism; downstream ``[:k]`` picks the k shots.
    """
    rows = load_jsonl_gz(path)
    rows = [r for r in rows if not r.get("truncated", False)]
    rows = [r for r in rows if r.get("think") and r.get("answer") and r.get("question")]
    if len(rows) < size:
        raise SystemExit(
            f"[build] shot pool {path} has only {len(rows)} valid rows (<{size}); "
            f"regenerate via rep_core/prepare_shot_pool.py"
        )
    sampled = rows[:size]
    return [
        {"q": r["question"], "r": r["think"], "a": r["answer"], "src_idx": r.get("id")}
        for r in sampled
    ]


def make_tokenizer(model_id: str = RENDER_MODEL_ID):
    """The Qwen3 family shares one chat template — render once with the 8B tokenizer."""
    from transformers import AutoTokenizer  # lazy import
    return AutoTokenizer.from_pretrained(model_id)


def render_chat(tok, defender_system: str, body: str,
                enable_thinking: bool = True,
                add_generation_prompt: bool = True) -> str:
    return tok.apply_chat_template(
        [
            {"role": "system", "content": defender_system},
            {"role": "user", "content": body},
        ],
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
        enable_thinking=enable_thinking,
    )


def build_inbox(
    *,
    cell_id: str,
    batch_name: str,
    test_rows: list[dict],
    tok,
    defender_system: str,
    body_fn,                     # (test_row) -> user-message body string
    out_path: Path,
    victim_key: str,
    max_new_tokens: int,
    max_input_tokens: int,
    temperature: float = 0.0,
    top_p: float = 1.0,
    top_k: int = -1,
    enable_thinking: bool = True,
    add_generation_prompt: bool = True,
    extra_meta_fn=None,          # (test_row) -> dict of extra meta_* fields
) -> dict:
    """Render one SGLang inbox JSONL (one row per test item) and return stats.

    Filename convention: ``<victim_key>__<batch_name>.jsonl`` (the farm keys the
    victim model off the filename prefix).
    """
    rows_out: list[str] = []
    n_skip = 0
    max_tok = 0
    for tr in test_rows:
        body = body_fn(tr)
        prompt = render_chat(tok, defender_system, body,
                             enable_thinking=enable_thinking,
                             add_generation_prompt=add_generation_prompt)
        n_tok = len(tok(prompt).input_ids)
        max_tok = max(max_tok, n_tok)
        if n_tok > max_input_tokens:
            n_skip += 1
            continue
        row = {
            "id": f"{tr['idx']}|cell={cell_id}",
            "prompt": prompt,
            "max_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "enable_thinking": enable_thinking,
            "add_generation_prompt": add_generation_prompt,
            "meta_cell_id": cell_id,
            "meta_test_idx": tr["idx"],
            "meta_n_input_tokens": n_tok,
        }
        if extra_meta_fn is not None:
            row.update(extra_meta_fn(tr))
        rows_out.append(json.dumps(row))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(rows_out) + ("\n" if rows_out else ""))
    print(f"[build] {cell_id:24s} n={len(rows_out):4d} skip={n_skip:3d} "
          f"max_in_tok={max_tok:6d}  -> {out_path.name}", flush=True)
    return {
        "cell_id": cell_id,
        "batch_name": batch_name,
        "n_rows": len(rows_out),
        "n_skipped": n_skip,
        "max_input_tokens": max_tok,
        "out_path": str(out_path),
    }


__all__ = [
    "OT_CONFIG", "OT_TEST_DIR", "SHOT_POOL_SIZE", "SHOT_POOL_SEED",
    "load_jsonl_gz", "load_ot500_test", "load_ot_shot_pool",
    "make_tokenizer", "render_chat", "build_inbox", "build_defender_system",
]
