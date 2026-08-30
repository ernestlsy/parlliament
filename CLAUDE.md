# tehpengagent — TikTok TechJam 2026, Track 2

Autonomous ML Research Agent for recommender systems. The agent must run the full MLE
loop by itself — read the problem, inspect data, engineer features, train and tune,
evaluate, reflect, iterate — on **KuaiRand-Pure**, and beat the organizers' FM baseline
on a hidden test set.

## Read these first

| File | When |
|---|---|
| `docs/BENCHMARK.md` | Task, metrics, splits, baselines, constraints, submission format, judging criteria, known dead ends |
| `docs/STATUS.md` | Where the system actually is, what is broken, open questions |
| `docs/DESIGN.md` | Agent architecture, invariants, decisions to revisit |
| `PROBLEM.md` | Official problem statement, verbatim. Read-only. |

## Where code goes

- `src/vibes/` — **active codebase.** All new work here.
- `src/ernest/` — frozen reference copy. Do not edit.
- `src/.tw/` — a second design + prototype. Idea source only, not a dependency.
- `src/kuairand-starter-kit/` — organizer-provided. `KuaiRand-Pure/` data already downloaded.

## Hard rules

1. **Never edit `evaluate.py`** (starter kit) or `src/vibes/agentic_recsys/evaluation.py`.
   The scoring convention is determined solely by `evaluate.py`. Agents must not be able
   to touch it either — that boundary is the system's core invariant.
2. **No external training data.** Only KuaiRand. No augmenting, joining, or pretraining
   on another dataset; no weights trained on these benchmarks' test labels. This is the
   one rule the organizers call hard.
3. **No hidden-test access.** Develop on train + validation only.
4. **Compare against the right baseline number.** Our runs score on validation, where the
   FM baseline is **0.6016** — not the 0.5946 test figure.
5. **Every iteration must be logged** with hypothesis, code diff, resulting metrics, and
   any error/recovery event. This is a graded deliverable, not just debug output.
6. **Count manual interventions.** Autonomy is 20% of the score and is measured by how
   few there were. If you intervene by hand during a run, record it.

## Facts that change what you build

- Metrics do **not** span [0,1]. Attainable range is 0.4753 (random) → 0.8645 (oracle).
  The baseline at 0.5946 already holds ~31% of it. Judge progress against 0.8645.
- Ranking is **within-user**, so any term constant within a user contributes exactly
  zero. User-side features only matter through crosses with item-side terms.
- More static features and more embedding capacity are both **measured dead ends** —
  the organizers published the ablations. Don't spend iterations rediscovering them.
- Scored result is the **converged** one (ε = 0.002, N = 3 on validation primary), i.e.
  the validation-best checkpoint at the stopping point. Not the peak, not the final.

## Conventions

- Python, numpy-first. The starter kit deliberately has no torch/pandas/sklearn
  dependency; heavier libraries are permitted but justify the cost — wall-clock is scored.
- Journals are append-only JSONL. Never rewrite history in place.
- Resume a run with the same `--workspace` and `--run-name`; the journal determines next
  ID, generation, parent archive, and stop state.
