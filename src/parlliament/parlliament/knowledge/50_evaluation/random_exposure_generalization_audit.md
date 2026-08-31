# Random-exposure generalization audit

## Summary and mechanism
Select checkpoints and hyperparameters **only** by the fixed official validation ranking protocol, then run this frozen secondary audit on a separately frozen random-exposure holdout. Score ordinary-log and random-exposure impressions separately with the same long-view label definition, user grouping, candidate-group construction, and ranking code. The comparison asks whether a validation gain persists when exposure is less determined by the historical recommender. KuaiRand inserts randomly sampled videos into normal feeds, creating observed feedback under an intervention that is useful for diagnosing exposure-policy dependence; it does not by itself establish the causal value of deploying a new policy. ([kuairand.com](https://kuairand.com/?utm_source=openai))

Assumptions: the exposure-regime indicator is correct; predictions use only information available before each impression; random items come from a documented pool; and every evaluated group contains a meaningful positive/negative label mix. Interpret the audit as external robustness evidence, not as a replacement validation target.

## When to use / avoid
**Use** after official validation selection when comparing popularity-heavy, causal-history, or debiased models; when a random-exposure log is frozen; and when ordinary and random impressions can be evaluated as distinct regimes.

**Avoid** using it for checkpoint selection, pooling regimes into one unstratified metric, or claiming unbiased policy value without logged propensities, candidate-pool/support definitions, and an approved off-policy protocol. KuaiRand records randomized interventions alongside sequential logs and rich feedback, but its random exposures are inserted into particular product contexts rather than representing every deployment distribution. ([kuairand.com](https://kuairand.com/?utm_source=openai))

## Requirements and implementation
1. Freeze: (a) canonical official-validation predictions and selected checkpoint, (b) random-exposure holdout row IDs, labels, candidate groups, and metric implementation.
2. Join each row by immutable impression/event ID to `exposure_regime`, `user_id`, `video_id`, timestamp, long-view label, and model score. Fail closed on duplicate joins, missing predictions, mixed timestamps, or regime reassignment.
3. Construct metric groups before inspecting model outputs. For GAUC, compute within-user AUC only for eligible user groups containing both label classes, then aggregate with the official group-weight rule; report eligible-user and eligible-impression coverage. For nDCG@5, rank only within the frozen candidate/impression group and report the number of groups with at least 5 candidates and the positive-label prevalence.
4. For every model, report ordinary and random GAUC/nDCG@5; absolute regime gaps (`random - ordinary`); paired per-group score differences where IDs overlap under the prescribed grouping; and matched support summaries: users, videos, impressions, groups, positives, candidate-set size, item-frequency quantiles, and cold/new-entity rates.
5. Keep preprocessing identical where feasible. Any regime-specific feature availability, filtering, label threshold, or candidate truncation must be logged as a protocol deviation, not silently normalized away.

Default: one fully frozen random holdout and one final selected checkpoint per model. If uncertainty reporting is permitted, use a fixed-seed user-level bootstrap (for example, 1,000 resamples); resample the same unit used by GAUC aggregation, preserve all impressions for sampled users, and report percentile intervals. This is an empirical stability summary, not a causal confidence interval.

## Starting configuration and expected effects
Start with the official long-view threshold, official GAUC/nDCG@5 code, all eligible groups, and no reweighting. Run the audit at the exact selected checkpoint; do not tune against its result. As a sensitivity analysis only, pre-register a minimum-candidate-size filter of 2 for GAUC and 5 for nDCG@5, plus an item-support restriction to videos represented in both regimes; report each restriction’s retained support.

A popularity/exposure-aligned model may retain strong ordinary-log ranking but lose relative GAUC or nDCG@5 on random exposure if historical exposure frequency acted as a shortcut. A debiased or causal-history variant may narrow that regime gap, but can also lose on both regimes if its correction adds variance or removes useful preference signal. No direction or magnitude should be presumed; random data are sparse relative to ordinary exposure in KuaiRand, so nDCG@5 can be especially noisy when candidate groups are small. ([kuairand.com](https://kuairand.com/?utm_source=openai))

## Diagnostics and risks
- **Leakage:** histories, popularity counts, normalizers, target encodings, and label-derived aggregates must be prefix-only relative to the impression timestamp. Check that no random-holdout row or future event contributes to any feature fit.
- **Support shift:** a random-regime decline concentrated in rare videos, unseen users, or small groups may reflect support mismatch rather than broad exposure-bias sensitivity. Inspect the matched-support analysis before attributing cause.
- **Group corruption:** unusually high nDCG@5, many singleton groups, or GAUC coverage changes across models commonly signal candidate-group or join errors.
- **Regime contamination:** if an impression can be both ordinary and randomized, define precedence from the raw intervention flag and exclude ambiguous rows with counts reported.
- **Compute:** cache one prediction artifact keyed by model/checkpoint/impression ID; auditing should be joins, sorting within frozen groups, and metric reductions—not retraining.

Diagnostic signature: a large ordinary-versus-random gap that survives shared-user/shared-video support restriction is stronger evidence of exposure-regime sensitivity than a gap that vanishes after matching. It remains descriptive evidence, not proof that any individual feature caused the gap.

## Cheapest check and clean experiment
**Cheap train-only check:** before touching either holdout, split training impressions by the recorded regime using a time-safe prefix. Compare long-view prevalence, user activity, video frequency, candidate-set size, and score distributions from a model trained without these audit labels. Large covariate or support differences warn that raw cross-regime metrics will be hard to interpret.

**Clean single-variable experiment:** hold architecture, seed set, training rows, feature timestamps, objective, checkpoint-selection rule, and prediction code fixed. Change exactly one treatment—for example, enable versus disable prefix-smoothed popularity features. Select each variant on canonical validation, then compare its frozen ordinary and random audit deltas and matched-support deltas. Do not choose the treatment, clipping, or checkpoint after seeing random-holdout results.

## Related cards and sources
Related cards: `dataset.random_exposure_log`, `dataset.interaction_log_schema`, `dataset.inventory_and_splits`, `task.kuairand_ranking`, `task.experiment_protocol`, `task.leakage_policy`, `task.prediction_artifact`, `evaluation.within_user_metrics`, `evaluation.frozen_candidate_group_integrity_audit`, `evaluation.stratified_temporal_population_evaluation`, `features.causal_behavior_history_features`, `features.train_prefix_smoothed_popularity_features`.

Primary source: Gao, C., Li, S., Zhang, Y., Chen, J., Li, B., Lei, W., Jiang, P., and He, X. (2022), *KuaiRand: An Unbiased Sequential Recommendation Dataset with Randomly Exposed Videos*, CIKM ’22, DOI: `10.1145/3511808.3557624`. ([kuairand.com](https://kuairand.com/?utm_source=openai))

### Audited web sources

- KuaiRand | An Unbiased Sequential Recommendation Dataset with Randomly Exposed Videos: <https://kuairand.com/?utm_source=openai>
