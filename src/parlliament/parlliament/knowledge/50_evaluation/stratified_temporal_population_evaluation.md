# Temporal and population-slice ranking dashboard

## Summary and mechanism
Evaluate one **frozen, canonical validation prediction artifact** both in aggregate and in a small set of **predeclared, scoring-time-valid slices**. Report GAUC and nDCG@5 by: validation-time block; user-history depth; train-prefix user and video frequency buckets; displayed candidate-group size; and candidate-group label composition (zero-positive, mixed, all-positive).

The purpose is diagnostic rather than subgroup optimization: an aggregate gain can conceal a late-period regression, cold-entity failure, or a gain driven only by degenerate groups. Group AUC is naturally aligned with user-grouped ranking evaluation; DIN formalized GAUC as an average of per-user AUC values, weighted by impression count. ([arxiv.org](https://arxiv.org/pdf/1911.07698?utm_source=openai)) Temporal availability matters particularly for popularity-derived quantities: a popularity baseline can be materially misrepresented if counts use interactions that occurred after the recommendation time. ([arxiv.org](https://arxiv.org/abs/2005.13829?utm_source=openai))

Assumptions: each displayed candidate group is intact; labels and candidate membership are frozen; timestamps define the validation order; and all slice keys are computable without future interactions or outcomes.

## When to use / avoid
**Use** when comparing close models, checking robustness across late validation blocks, investigating sparse users/videos, or determining whether a nDCG@5 movement comes from rankable mixed-label groups.

**Avoid** using the dashboard to search dozens of post-hoc cuts, to declare a winner from a tiny slice, or to construct buckets from validation-period cumulative frequency, future labels, post-impression engagement, or model outputs.

## Requirements and implementation
1. Materialize one row per candidate impression with `group_id`, `user_id`, `video_id`, impression timestamp, binary long-view label, score, and split/model/checkpoint identifiers.
2. Join immutable train-prefix counts: `user_train_count` and `video_train_count`. Count only events strictly before the training cutoff; if features are intended to emulate online scoring, use counts available immediately before each impression instead.
3. Define slices before inspecting model comparison results:
   - **Time:** 4–8 contiguous validation blocks with approximately equal impressions; retain chronological order.
   - **User history:** `0`, `1–4`, `5–19`, `20–99`, `100+` prior train-prefix events; merge adjacent bins if support is weak.
   - **User/video frequency:** `0`, `1–4`, `5–19`, `20–99`, `100+` train-prefix events.
   - **Group size:** `1`, `2–5`, `6–20`, `21–100`, `101+` candidates.
   - **Label composition:** zero-positive, mixed, and all-positive, determined within the displayed group.
4. Compute GAUC only over groups or users containing both label classes. Compute nDCG@5 per displayed group, using the benchmark’s declared handling of groups with no positives; report the denominator explicitly.
5. For every cell, show metric, eligible-group count, impression count, positive count/rate, and a 95% uncertainty interval. Prefer a cluster bootstrap resampling users, with all of each sampled user’s groups retained; use 500–2,000 replicates initially. If repeated users are uncommon, resample candidate groups and label the interval accordingly.
6. Publish aggregate metrics beside slices, plus the model-to-baseline delta and uncertainty interval for each prespecified cell.

## Starting configuration and expected effects
Start with five frequency/history buckets and four chronological blocks. Require at least 100 eligible mixed-label groups for a displayed GAUC cell and at least 100 groups for a displayed nDCG@5 cell; otherwise merge neighboring bins or mark the estimate exploratory. These are practical defaults, not universal statistical thresholds.

Expect aggregate GAUC and nDCG@5 to be more stable than fine-grained cells. Zero-positive and all-positive groups carry little or no within-group ranking discrimination: they should be reported for prevalence and data-quality monitoring, but should not drive a ranker choice. A model that improves only warm, frequent-entity cells may raise aggregate metrics while increasing cold-start deployment risk. Conversely, a small aggregate loss may be acceptable if it removes a material late-time or cold-population regression; this is a product decision, not a conclusion supplied by the dashboard.

## Diagnostics and risks
- **Late-block decline with stable early blocks:** temporal drift, stale features, changed inventory, or an invalid split boundary.
- **Cold video/user decline but warm gains:** ID memorization, insufficient metadata, or train-prefix popularity dependence.
- **Only large groups improve in nDCG@5:** inspect whether small groups dominate traffic or have different label prevalence.
- **GAUC moves but nDCG@5 does not:** score separation changed below the top ranks, or gains occur in groups where the top-five cutoff is insensitive.
- **nDCG@5 moves only in zero/all-positive groups:** likely an evaluator convention, group-integrity error, or denominator mismatch; do not interpret as ranking quality.
- **Very wide intervals or sign reversals across bootstrap runs:** insufficient support, correlated repeated impressions, or excessive slicing.

Leakage hazards include computing entity frequencies using the full dataset, assigning time buckets after filtering on outcomes, letting candidate groups fragment during joins, and using post-impression fields. Computational cost is normally low because inference is reused; bootstrap evaluation can be expensive, so cache per-group sufficient statistics where metric definitions permit.

## Cheapest check and clean experiment
**Cheap train-only check:** before training any new model, create the bucket keys from train-prefix logs and tabulate validation support, mixed-label-group counts, label prevalence, group sizes, and cold-user/video overlap for each planned slice. Fail the dashboard build if a group is duplicated, candidate count changes after joins, timestamps cross the split cutoff, or a slice key depends on validation outcomes.

**Clean single-variable experiment:** compare the current model against exactly one variant—for example, adding train-prefix smoothed video-popularity features—using the identical training seed set, frozen split, candidate groups, evaluator, and checkpoint rule. Pre-register aggregate GAUC, aggregate nDCG@5, the final time block, and cold-video buckets as decision cells. Select neither model from an unplanned slice; treat other cells as diagnosis.

## Related cards and sources
Related IDs: `dataset.inventory_and_splits`, `dataset.population_and_pair_shift`, `evaluation.within_user_metrics`, `evaluation.frozen_candidate_group_integrity_audit`, `task.experiment_protocol`, `task.leakage_policy`, `task.prediction_artifact`, `features.train_prefix_smoothed_popularity_features`.

Primary sources: Zhou et al., *Deep Interest Network for Click-Through Rate Prediction*, KDD 2018, DOI: 10.1145/3219819.3219823 (GAUC in grouped CTR evaluation); Ji, Sun, Zhang, and Li, *A Re-visit of the Popularity Baseline in Recommender Systems*, arXiv:2005.13829 (time-aware popularity evaluation). ([arxiv.org](https://arxiv.org/abs/2005.13829?utm_source=openai))

### Audited web sources

- A Troubling Analysis of Reproducibility and Progress in: <https://arxiv.org/pdf/1911.07698?utm_source=openai>
- A Re-visit of the Popularity Baseline in Recommender Systems: <https://arxiv.org/abs/2005.13829?utm_source=openai>
