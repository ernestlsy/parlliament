# Frozen displayed-candidate evaluation audit

## Summary and mechanism
Official ranking evaluation is defined over the **displayed candidate groups supplied for validation**, not over a reconstructed catalog or newly sampled negatives. Treat each displayed group as an immutable query/impression set: preserve its membership, canonical row order, labels, and group-to-user mapping; score every row exactly once; then rank only within that group. This makes GAUC and nDCG@5 comparable across models because their denominators and candidate universes are identical.

For binary long-view labels, compute group AUC from all positive–negative pairs in each eligible group, assigning ties 0.5 credit. Aggregate using the competition or product’s declared weighting rule; if none is specified, record the chosen rule explicitly rather than silently changing it. For nDCG@5, sort descending by score with canonical row position as the deterministic final tie-breaker, compute DCG@5, divide by the group’s IDCG@5, and average only across groups with IDCG@5 > 0. nDCG is a normalized cumulative-gain measure whose value depends on the judged items and their ranks. ([researchportal.tuni.fi](https://researchportal.tuni.fi/en/publications/cumulated-gain-based-evaluation-of-ir-techniques?utm_source=openai))

Do not substitute sampled or synthetic negatives into the official scorer. Sampled ranking metrics can fail to preserve model comparisons relative to their exact counterpart, including in expectation; therefore they are a different measurement problem, not an acceleration of the fixed displayed-set metric. ([research.google](https://research.google/pubs/on-sampled-metrics-for-item-recommendation/?utm_source=openai))

## When to use / avoid
**Use** for official within-user GAUC or nDCG@5, prediction-artifact validation, model comparisons where batching, distributed inference, candidate-conditioned features, or ranking heads can reorder/drop/duplicate impressions.

**Avoid** only when the benchmark explicitly defines another candidate universe, such as full-catalog retrieval. A sampled-negative metric may be useful as a separately named development diagnostic, but is not interchangeable with this result.

## Requirements and implementation
Require a canonical validation table with immutable `(group_id, canonical_row_index)` keys, user IDs, long-view labels, and one prediction per row. If candidate IDs are available, retain them as an additional audit key; do not use them to collapse repeated impressions.

1. Materialize the supplied validation groups once, before model inference. Store row count, ordered key hash, group sizes, label counts, and split/version identifier.
2. Emit predictions keyed by the canonical row key. Reject missing keys, extra keys, duplicate keys, non-finite scores, or a changed ordered-key hash.
3. Left-join predictions onto canonical rows; never inner-join from predictions, which can conceal omissions.
4. Assert group completeness: observed scored rows equal canonical rows for every group. Assert each group maps to one user unless multi-user groups are part of the specification.
5. Audit labels before scoring: report group count, rows, positive rate, all-negative groups, all-positive groups, singleton groups, and groups eligible for pairwise GAUC.
6. Apply a documented deterministic sort: `score DESC, canonical_row_index ASC`. Do not let framework-specific unstable sorting choose top-5 ties.
7. Save a machine-readable audit report alongside metrics, including exclusion counts and the metric denominator actually used.

Recommended hard defaults: fail closed on any key-set or duplicate mismatch; require 100% score coverage; reject NaN/Inf; use zero tolerance for row-count discrepancies; use canonical row index only for score ties. There is no tuning range for integrity checks: relaxing these checks changes the evaluated object. For exploratory diagnostics only, permit a separate report with score-coverage thresholds of 99.9–100%, but never publish an official metric when coverage is incomplete.

## Starting configuration and expected effects
Start with a single canonical scorer shared by every model and checkpoint. Compute both a macro group mean and the required official aggregate in the audit artifact, clearly labeling only one as official. Exclude groups without both label classes from pairwise AUC and report their count; exclude zero-IDCG groups from nDCG and report that denominator. Do not convert undefined per-group values to zero without an explicit benchmark rule.

Integrity fixes can move GAUC or nDCG@5 in either direction; no universal magnitude should be expected. Restoring omitted hard candidates commonly makes ranking harder, while deduplicating accidental rows can either remove repeated positives or repeated negatives. The key expected effect is **valid comparability**, not guaranteed improvement. Because nDCG@5 emphasizes the top ranks, it is especially sensitive to candidate loss, duplicate rows, and non-deterministic score ties; GAUC is especially sensitive to the set of positive–negative pairs included.

## Diagnostics and risks
**Diagnostic signatures**
- Different row counts or ordered-key hashes across models: dataloader filtering, shard loss, or an invalid join.
- Coverage below 100% concentrated in large groups: truncation, maximum-sequence limits, or distributed gather failure.
- More prediction rows than canonical rows: repeated inference, retry concatenation, or many-to-many joins.
- Abrupt nDCG@5 movement with nearly unchanged GAUC: top-k tie handling, group reorder, or a small number of missing high-scored rows.
- Abrupt GAUC movement with similar nDCG@5: altered pair denominators, label parsing, or inclusion/exclusion of single-class groups.
- Strong score shifts by canonical row position: accidental positional leakage or sorting/order features entering the model.

Prevent leakage by constructing validation features strictly from each row’s allowed train/history prefix and by keeping labels out of candidate features, normalization fits, target encoders, and checkpoint selection inputs. The audit itself is inexpensive relative to inference: sorting costs roughly `O(sum_g |g| log |g|)` and pairwise GAUC should use a rank-based implementation rather than explicitly materializing all positive–negative pairs. Never use validation audit failures to patch labels or candidate membership; repair the pipeline and regenerate predictions.

## Cheapest check and clean experiment
**Cheap train-only check:** take a held-out train-prefix slice, freeze its canonical groups, run inference twice with different batch sizes/shard counts, and compare ordered-key hashes, row counts, score coverage, duplicate counts, and final metrics. Exact equality is expected when inference is deterministic; otherwise key integrity must still match exactly and score differences should be separately explained.

**Clean single-variable experiment:** hold model weights, validation rows, scorer, tie-break, and aggregation rule fixed. Compare (A) the canonical displayed groups with (B) an auxiliary sampled-negative evaluation, using the same predictions wherever possible. Report them as distinct metrics and compare model ordering, not merely absolute values. This isolates the effect of changing the candidate universe; it must not replace the official result. The empirical literature specifically cautions that naive sampled metrics need not retain exact-metric model order. ([research.google](https://research.google/pubs/on-sampled-metrics-for-item-recommendation/?utm_source=openai))

## Related cards and sources
Related cards: `evaluation.within_user_metrics`, `task.prediction_artifact`, `task.experiment_protocol`, `task.leakage_policy`, `dataset.interaction_log_schema`, `dataset.inventory_and_splits`, `training.group_complete_stratified_minibatching`.

Primary sources: Järvelin and Kekäläinen, *Cumulated Gain-Based Evaluation of IR Techniques* (2002), DOI: `10.1145/582415.582418`. ([researchportal.tuni.fi](https://researchportal.tuni.fi/en/publications/cumulated-gain-based-evaluation-of-ir-techniques?utm_source=openai)) Krichene and Rendle, *On Sampled Metrics for Item Recommendation* (KDD 2020). ([research.google](https://research.google/pubs/on-sampled-metrics-for-item-recommendation/?utm_source=openai))

### Audited web sources

- Cumulated Gain-based Evaluation of IR Techniques - Tampere University Research Portal: <https://researchportal.tuni.fi/en/publications/cumulated-gain-based-evaluation-of-ir-techniques?utm_source=openai>
- On Sampled Metrics for Item Recommendation: <https://research.google/pubs/on-sampled-metrics-for-item-recommendation/?utm_source=openai>
