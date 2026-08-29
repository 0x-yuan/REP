"""Downstream QA utility eval on three non-math reasoning categories:
StrategyQA (commonsense), ProntoQA (symbolic), HotpotQA (multi-hop, open-book).

Pure-python scoring lives in ``scoring.py``; the eval-set builder and the
generation runner are thin CLIs on top of it.
"""
