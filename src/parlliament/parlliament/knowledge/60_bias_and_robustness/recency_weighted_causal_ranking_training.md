# Recency-weighted causal ranking training

## Summary and mechanism

Apply a **group-level, bounded exponential recency weight** to every complete displayed candidate group in the training prefix. For a group with impression timestamp t, training cutoff T, half-life h, and lower bound w_min, use:

`raw_weight = max(w_min, 2^(-(T - t) / h))`.

Multiply the loss for every candidate in that displayed group by the same `raw_weight`; do not change labels, remove candidates, or weight clicked and unclicked candidates differently. Then normalize weights so the average effective training weight is 1, preferably after the existing per-user normalization. This keeps the original within-group ranking and exposure comparison intact while assigning more optimization influence to recent impressions.

The motivation is temporal nonstationarity: user preferences, catalog composition, and item popularity can change over time. Temporal recommender research explicitly identifies these changing factors and cautions that simply discarding older observations can lose useful signal. ([yuzhang-teaching.github.io](https://yuzhang-teaching.github.io/CSCE670-F25/Koren09.pdf?utm_source=openai)) Example age is also a meaningful temporal signal in production recommendation work, where it was used to help represent time-dependent item behavior. ([research.google.com](https://research.google.com/pubs/archive/45530.pdf?utm_source=openai))

This is **not** exposure debiasing by itself. It only changes the empirical training distribution over logged impressions. Retain complete displayed groups because implicit-feedback logs have dynamic missingness and unobserved outcomes are not automatically valid negatives outside the actually exposed candidate set. ([papers.neurips.cc](https://papers.neurips.cc/paper_files/paper/2018/hash/8d9766a69b764fefc12f56739424d136-Abstract.html?utm_source=openai))

## When to use / avoid

Use when chronological validation shows recent-period GAUC or nDCG@5 degradation, and reliable timestamps indicate that the serving population, inventory, or engagement process has shifted. Use timestamps relative to the fixed training cutoff only; validation and test timestamps must never influence a training example's weight.

Avoid when the official split is effectively i.i.d., when temporal slices show no practical degradation, or when the newest portion of training data is too sparse to support a stronger recent-data emphasis. Also avoid the method until timestamp semantics are resolved: event time, logging time, delayed-label time, and backfilled-ingestion time are not interchangeable.

## Requirements and implementation

Required fields are: impression timestamp, training cutoff timestamp, displayed candidate-group identifier, and long-view label. Build each optimization record from the full displayed group, including non-long-view candidates. Compute group age in a single timezone and unit, clamp negative ages to zero only after asserting that no timestamp exceeds T except for documented clock skew.

Implementation steps:

1. Freeze the chronological split and define T as the final eligible training impression time.
2. Assign one timestamp and one weight to each displayed group; all candidates in that group receive exactly the same multiplier.
3. Use `w_g = max(w_min, 2^(-age_days / h_days))`.
4. Apply `w_g` outside the group loss, after any within-group aggregation. If using user-normalized BCE, normalize user contributions first and then rescale all weights by a global constant so mean effective weight is 1.
5. Log unnormalized and normalized weights, weighted label prevalence, effective sample size, and weight mass by time bucket.
6. Keep the baseline sampler, optimizer schedule, model architecture, candidate construction, checkpoint rule, and all non-recency features unchanged.

Do not use recency weights to select only positive events, to drop old non-clicked impressions, or to recompute historical candidate groups from a later catalog snapshot. Those choices change the comparison set or introduce leakage rather than merely reweighting the existing training objective.

## Starting configuration and expected effects

Treat these as conservative engineering defaults, not universal research findings:

- Half-life grid: 14, 30, 60, and 120 days; predeclare the grid before evaluation.
- Minimum weight: 0.10 or 0.20. A positive floor preserves long-tail entities and recurring preferences that are sparse in recent data.
- Select one configuration by chronological validation GAUC, using nDCG@5 as a guardrail; if the metrics disagree, inspect temporal and entity-frequency slices before choosing.
- Renormalize total training weight to preserve approximately the baseline loss scale and avoid conflating recency tuning with a learning-rate change.

Expected effect: if genuine drift exists, recent-slice GAUC and nDCG@5 may improve because the fitted ranker gives more influence to current interactions. Aggregate metrics can remain flat or decline if the older distribution is still representative, if recency mostly captures transient popularity, or if recent data are noisier. Do not claim an improvement magnitude without a controlled result on the target data. Earlier work on temporal collaborative filtering supports modeling temporal variation, but it does not establish a universal best half-life or guarantee gains for this loss and dataset. ([yuzhang-teaching.github.io](https://yuzhang-teaching.github.io/CSCE670-F25/Koren09.pdf?utm_source=openai))

## Diagnostics and risks

**Leakage checks.** Assert `impression_time <= T` for every training group; compute all ages from T, not from wall-clock training time; and ensure that group membership, labels, user histories, and item features are frozen at each impression time. Audit for duplicated groups crossing the train-validation boundary.

**Weight-collapse signature.** If a small recent window carries most normalized weight, effective sample size drops sharply, training loss becomes volatile, and performance on low-frequency users or older-but-still-active items falls. Increase h or w_min.

**Popularity-chasing signature.** If recent overall nDCG@5 rises but catalog concentration rises sharply and long-tail or repeat-preference slices fall, the model may be tracking short-lived popularity rather than durable relevance. Increase the floor, lengthen h, and inspect item-age and popularity-stratified results.

**No-drift signature.** If chronological slices are stable and all half-lives perform within noise of the unweighted baseline, select the baseline for simplicity.

**Confounding signature.** If a product-policy, position, or candidate-generator change coincides with the high-weight period, the ranker can learn changed exposure patterns rather than changed preference. Recency weighting does not correct this; use complete logged groups and consider an exposure-debiased method when suitable propensities or randomized exposure evidence exists. Dynamic missingness in implicit feedback is itself a documented modeling concern. ([papers.neurips.cc](https://papers.neurips.cc/paper_files/paper/2018/hash/8d9766a69b764fefc12f56739424d136-Abstract.html?utm_source=openai))

Compute cost is negligible: one scalar per group and no additional model pass. The practical cost is experiment multiplication across the half-life grid and added monitoring of weighted-population shift.

## Cheapest check and clean experiment

**Cheapest train-only check:** before fitting any model, make weekly or monthly tables of raw group count, normalized weight mass, long-view prevalence, unique users, unique items, and effective sample size. Verify that all candidates in a group have identical weight, global mean normalized weight is 1, no future group is present, and no single time bucket unexpectedly dominates. This catches cutoff, unit, and join errors without using validation labels for tuning.

**Clean single-variable experiment:** train the exact same seed set and training schedule for: unweighted baseline, then half-lives 14, 30, 60, and 120 days with fixed `w_min = 0.10`. Predeclare selection as best chronological-validation GAUC, break ties with nDCG@5, and report aggregate plus recent/older temporal slices. Hold candidate groups, negative construction, user normalization, feature snapshots, and checkpoint selection protocol fixed. After choosing one half-life, run a confirmation retrain on fresh seeds; only then test a second floor value if the selected model shows weight-collapse or long-tail regression.

## Related cards and sources

Related cards: `dataset.interaction_log_schema`, `dataset.inventory_and_splits`, `dataset.population_and_pair_shift`, `dataset.random_exposure_log`, `evaluation.within_user_metrics`, `task.experiment_protocol`, `task.leakage_policy`, `objective.user_normalized_binary_cross_entropy`, `training.group_complete_stratified_minibatching`, `evaluation.frozen_candidate_group_integrity_audit`, `evaluation.stratified_temporal_population_evaluation`, `robustness.clipped_ips_within_group_rank_loss`, `robustness.doubly_robust_exposure_debiased_ranking`.

Primary sources: Yehuda Koren, “Collaborative Filtering with Temporal Dynamics,” KDD 2009, DOI: 10.1145/1557019.1557072. ([dblp.org](https://dblp.org/rec/conf/kdd/Koren09.html?utm_source=openai)) Paul Covington, Jay Adams, and Emre Sargin, “Deep Neural Networks for YouTube Recommendations,” RecSys 2016. ([research.google.com](https://research.google.com/pubs/archive/45530.pdf?utm_source=openai)) Dawen Liang et al., “Modeling Dynamic Missingness of Implicit Feedback for Recommendation,” NeurIPS 2018. ([papers.neurips.cc](https://papers.neurips.cc/paper_files/paper/2018/hash/8d9766a69b764fefc12f56739424d136-Abstract.html?utm_source=openai))

### Audited web sources

- Collaborative Filtering with Temporal Dynamics: <https://yuzhang-teaching.github.io/CSCE670-F25/Koren09.pdf?utm_source=openai>
- Deep Neural Networks for YouTube Recommendations: