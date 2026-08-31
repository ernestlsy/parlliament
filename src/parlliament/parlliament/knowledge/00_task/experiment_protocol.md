# Counted-experiment and convergence protocol

## Summary and mechanism
ParLLiaMent has a scarce budget of at most 50 successfully scored experiments. The selected seed scaffold
is unscored parent 0. It may be the fresh simple scaffold or the optional KuaiRand FM baseline code,
but neither selection inserts a measured baseline result. Train-only screening, planning, failed
retries, and abandoned attempts do not receive experiment IDs or consume the counted budget.

## When to use / avoid
Use this card whenever prioritizing hypotheses or interpreting lineage. Prefer one high-information,
bounded change over broad searches. Avoid citing parent 0 as measured evidence, treating an
abandoned attempt as an experiment, or spending official experiments on basic schema discovery.

## Requirements and implementation
Only a successful validation score atomically becomes `experiment_N` and enters the Journal.
Generated candidates must reference parent 0 or an actually scored experiment available in the
current archive. One tournament winner proceeds to implementation per generation, even when draft
planning considered a larger candidate portfolio.

## Starting configuration and expected effects
Before an official experiment, use the cached train-only temporal screen, literature, prior metrics,
and segment diagnostics to state one exact ablation and an expected effect on both GAUC and nDCG@5.
Reserve official scoring for changes with a plausible primary gain and a clean fallback.

## Diagnostics and risks
Every scored experiment with primary score `x` establishes a target of `x + 0.002`. ParLLiaMent stops if
all three subsequent scored experiments are strictly below that target, so the check requires at
least four scored experiments. Reaching the target in any of those three experiments prevents that
anchor from causing convergence. Abandoned attempts are absent from this sequence. The globally
highest-scoring validation experiment is selected for submission even when a later, lower-scoring
experiment triggers convergence. Repeated implementation failures waste wall-clock time despite
being uncounted.

## Cheapest check and clean experiment
Use profiling, train-only temporal holdout screening, contract checks, and code validation before
training. Then change the smallest coherent set of files needed for the winning hypothesis and
compare it against its declared scored parent.

## Related cards and sources
See `task.kuairand_ranking`, `task.prediction_artifact`, and
`evaluation.within_user_metrics`. The authoritative source is `parlliament_mle_plan.md` plus the
implemented Journal and Overseer convergence checks.
