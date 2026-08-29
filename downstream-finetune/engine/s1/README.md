# Vendored trainer (from simplescaling/s1)

`train/sft.py` is the s1 SFT trainer (Apache-2.0, see `LICENSE`), vendored so the
distillation recipe is frozen. Only `train/` is kept; upstream data-generation
and eval folders are not part of this release.
