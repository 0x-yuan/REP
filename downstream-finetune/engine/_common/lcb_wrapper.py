"""Wrapper around `lcb_runner.runner.main` that monkey-patches the
hardcoded gated tokenizer reference.

lcb_runner's `prompts/code_generation.py` hardcodes:
    AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct", ...)
inside the LMStyle.LLaMa3 prompt formatter. That repo is HF-gated
separately from `meta-llama/Llama-3.1-8B-Instruct` (which we already have
access to). To avoid the GatedRepoError without applying for a second HF
gate, we redirect the request to a local path (Llama-3 / 3.1 share the
same chat template, so this is correctness-preserving).

Usage:
    LCB_LOCAL_TOKENIZER_PATH=<ckpt_path> python -m _common.lcb_wrapper \
        --model meta-llama/Llama-3.1-8B-Instruct --local_model_path <ckpt_path> ...
"""
from __future__ import annotations

import os
import sys


def _install_tokenizer_redirect() -> None:
    """Patch AutoTokenizer.from_pretrained so the gated upstream id is
    rewritten to the local ckpt path. No-op if env var is unset."""
    local_path = os.environ.get("LCB_LOCAL_TOKENIZER_PATH")
    if not local_path:
        return
    from transformers import AutoTokenizer

    _orig = AutoTokenizer.from_pretrained

    def _patched(name_or_path, *args, **kwargs):
        name = str(name_or_path) if name_or_path is not None else ""
        if "Meta-Llama-3-8B-Instruct" in name or "meta-llama/Meta-Llama-3-8B" in name:
            print(
                f"[lcb_wrapper] redirecting AutoTokenizer({name_or_path!r}) -> {local_path!r}",
                file=sys.stderr,
                flush=True,
            )
            name_or_path = local_path
        return _orig(name_or_path, *args, **kwargs)

    AutoTokenizer.from_pretrained = _patched


def main() -> None:
    _install_tokenizer_redirect()
    # Defer the heavy import so the patch is in place first.
    from lcb_runner.runner.main import main as _lcb_main

    _lcb_main()


if __name__ == "__main__":
    main()
