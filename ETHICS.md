# Responsible-use statement

REP (Reasoning Exposure Prompting) is a **dual-use** technique. It shows that
interface-level hiding of a model's internal reasoning trace does not prevent a
user from recovering a close approximation of that trace through prompting
alone, and that the recovered traces are useful distillation supervision.

We release this code, data and checkpoints so that:

- model providers can measure their own exposure and evaluate defenses (see
  `eval/defenses/` — the two published black-box prompt defenses we tested do
  not stop REP);
- the research community can reproduce every number in the paper and build on
  the evaluation protocol (`eval/`, `results/`).

## What we ask of users

1. **Do not use REP to extract reasoning from a commercial service in violation
   of its terms of service.** All released data and checkpoints derive from
   open-weight models.
2. **Do not redistribute exposed traces as training data for a competing
   product.** The released corpora are provided for reproducibility of the
   paper's distillation-value study only.
3. Cite the paper (see `CITATION.cff`) and keep this file with any derivative.

## Not part of this release

- The victims' internal (oracle) trace corpora used for the reference students.
- Corpora harvested for ablations only (answer-only / summary controls), and
  intermediate harvests.
- Any per-provider API keys, prompts that target a specific deployed product's
  system prompt, or automation beyond what is needed to reproduce the paper.

Questions about responsible use: open an issue or contact the corresponding
author listed in the paper.
