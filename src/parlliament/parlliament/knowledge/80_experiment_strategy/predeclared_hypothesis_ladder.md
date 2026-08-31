# Predeclared hypothesis ladder with stop and rollback rules

## Summary and mechanism

Build a short, ordered sequence of **single, falsifiable changes** from one frozen reference configuration. Each rung states: parent run ID; exact configuration diff; one hypothesized mechanism; expected direction for the predeclared composite metric and its GAUC/nDCG@5 components; resource budget; confirmation rule; and rollback action. Train rung *k* only if rung *k−1* meets its rule. Otherwise retain the parent and spend the next run on a change aimed at a different diagnosed failure mode.

The mechanism is decision discipline under adaptive experimentation: it limits untracked degrees of freedom, preserves comparable artifacts, and makes negative results actionable. This does not make a small validation set unbiased or eliminate multiple-testing risk; it makes the sequence and its decisions auditable. Recommender-systems studies have documented reproducibility and comparison problems when protocols, baselines, and implementations are not controlled consistently. ([arxiv.org](https://arxiv.org/abs/2102.00482?utm_source=openai))

Assumptions: the reference model is trainable; candidate changes can be isolated; validation candidate groups and labels are frozen; and GAUC and nDCG@5 are computed by the same code for every rung.

## When to use / avoid

**Use when** official run IDs, accelerator time, or reviewer attention are scarce; competing changes span features, architecture, objectives, or training; and a promotion/abandonment rationale must survive later audit.

**Avoid when** inexpensive local compute supports broad exploration, or the proposed method requires coupled changes that cannot be meaningfully ablated. In the latter case, declare the coupled bundle as one rung and explicitly state that its internal causal contributions are not identified.

## Requirements and implementation

Create an immutable ledger row before launch:

```text
ladder_id, rung, run_id, parent_run_id, git_commit, data_snapshot,
split_id, prediction_artifact_id, config_diff, hypothesis, mechanism,
primary_metric_definition, expected_GAUC_direction, expected_nDCG5_direction,
training_budget, seed_policy, acceptance_rule, rollback_rule, status
```

1. Freeze the data snapshot, temporal cutoff, group construction, feature-generation cutoff, evaluator version, candidate ordering policy, and validation prediction artifact schema.
2. Predeclare one composite metric, for example `M = w_g * normalized_GAUC + w_n * normalized_nDCG@5`, including fixed weights and normalization constants derived only from the reference or a prior locked protocol. Keep GAUC and nDCG@5 as mandatory reported components; never let a composite hide a material regression in either.
3. Set a **non-inferiority guardrail** for each component and an **improvement margin** for `M` before observing the rung. Do not claim a universal useful delta: the margin should reflect validation noise, business relevance, and the historical seed-to-seed variation of the reference.
4. Default to 3–6 rungs, one material diff per rung, and identical training-token/epoch/step budgets within a comparison. These are operational defaults, not research-derived optima.
5. Use fixed seeds for the cheapest screen; for a rung that clears, rerun the parent and child under the predeclared confirmation seed set. Record all runs, including failures and retries.
6. On rejection, roll back to the exact parent checkpoint/configuration, preserve the child artifact, label the hypothesized mechanism unsupported, and choose the next rung from a different diagnostic branch rather than quietly modifying the rejected change.

## Starting configuration and expected effects

Start with a locked reference such as `architecture.embedding_mlp_ranker` trained with `objective.user_normalized_binary_cross_entropy`, using `training.group_complete_stratified_minibatching` and the established split/evaluation protocol.

Example ladder:

1. **Feature rung:** add only `features.train_prefix_smoothed_popularity_features`. Hypothesis: a leakage-safe popularity prior improves broad discrimination; expected GAUC direction: up or neutral; nDCG@5: up or neutral.
2. **Objective rung:** from the accepted parent only, replace/add a predeclared ranking component using `objective.bce_lambdarank_ndcg5_hybrid`. Hypothesis: a top-rank-aligned signal helps nDCG@5 more directly; GAUC may improve, remain neutral, or decline.
3. **Architecture rung:** from the accepted parent only, add `architecture.candidate_conditioned_history_attention`. Hypothesis: candidate-specific history relevance improves top-of-list ordering; compute and overfitting risk increase.

These are directional hypotheses, not promised effect sizes. A GAUC gain with flat or lower nDCG@5 suggests better global ordering without better head ranking; an nDCG@5 gain with GAUC loss suggests concentrated top-rank trade-offs. Neither pattern alone establishes the proposed mechanism.

## Diagnostics and risks

**Leakage:** verify every feature is computable at the prediction timestamp using train-prefix data only; audit joins, aggregates, normalization statistics, target encodings, and cached embeddings. Reuse `task.leakage_policy` and `evaluation.frozen_candidate_group_integrity_audit`.

**Compute confounding:** a larger model, longer schedule, changed batch composition, or altered early stopping is not a single-variable architecture test. Hold budget fixed; if extra compute is intrinsic, make it part of the rung claim and report it.

**Adaptive overfitting:** repeated inspection of the same validation set can turn a ladder into a hidden sweep. The ledger, fixed artifacts, and confirmation reruns reduce this risk but do not replace a final untouched test evaluation.

**Diagnostic signatures:** training loss improves while both ranking metrics fall → optimization/objective mismatch or leakage-safe feature failure; GAUC rises but nDCG@5 falls → inspect score calibration, within-group rank errors, and top-5 candidate coverage; only one seed wins → treat as unconfirmed; metrics move with changed candidate-group counts → suspect evaluator or split drift.

## Cheapest check and clean experiment

**Cheap train-only check:** before a submitted run, execute one short deterministic training slice using the parent’s data snapshot and exactly the child config. Assert: config diff contains only the intended fields; parameter count and FLOPs match the declared change; no validation/test timestamps enter features; loss is finite; prediction rows align one-to-one with frozen candidate groups; and GAUC/nDCG@5 evaluators consume identical group IDs and candidates. Do not use this screen to promote a method.

**Clean experiment:** train the parent and one child with identical data, split, evaluator, checkpoints, maximum steps, batch construction, seed set, and prediction-artifact format. Change exactly one declared intervention. Accept only if the composite clears its predeclared margin, both component guardrails hold, and the confirmation reruns satisfy the same rule; otherwise restore the parent and open a new rung for a distinct failure hypothesis.

## Related cards and sources

Related IDs: `task.experiment_protocol`, `task.leakage_policy`, `task.prediction_artifact`, `evaluation.within_user_metrics`, `evaluation.frozen_candidate_group_integrity_audit`, `evaluation.stratified_temporal_population_evaluation`, `evaluation.probability_calibration_and_ranking_error_audit`, `efficiency.successive_halving_experiment_promotion`.

Primary sources: Bellogín and Said, *Improving Accountability in Recommender Systems Research Through Reproducibility*, arXiv:2102.00482; Ferrari Dacrema, Cremonesi, and Jannach, *Are We Really Making Much Progress? A Worrying Analysis of Recent Neural Recommendation Approaches*, arXiv:1907.06902; Ferrari Dacrema, Cremonesi, and Jannach, *A Troubling Analysis of Reproducibility and Progress in Recommender Systems Research*, arXiv:1911.07698. These motivate controlled, reproducible comparison; the ladder’s particular defaults are practical policy choices rather than experimentally universal thresholds. ([arxiv.org](https://arxiv.org/abs/2102.00482?utm_source=openai))

### Audited web sources

- Improving Accountability in Recommender Systems Research Through Reproducibility: <https://arxiv.org/abs/2102.00482?utm_source=openai>
