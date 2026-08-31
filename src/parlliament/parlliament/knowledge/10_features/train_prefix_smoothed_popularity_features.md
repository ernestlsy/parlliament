# Train-prefix smoothed popularity and affinity statistics

## Summary and mechanism
Create features from interactions that occurred **strictly before** each scored impression time. For a binary target \(y\), maintain timestamped sufficient statistics such as item/video exposures and positives, creator/category exposures and positives, and user-by-creator or user-by-category exposures and positives. Convert sparse rates into smoothed estimates:

\[
\hat p_g = \frac{c_g^+ + \alpha p_0}{c_g + \alpha}
\]

where \(c_g\) is prior eligible exposure count for group \(g\), \(c_g^+\) is prior positive count, \(p_0\) is a train-prefix prior, and \(\alpha\) is pseudo-count strength. Use the same construction for user–attribute affinity groups. Add transformed support features, e.g. `log1p(c_g)`, age since first observation, and recent-window variants.

The key assumption is that historical interactions and, preferably, historical exposures are available as-of the impression. Ordered/causal target-statistic construction is specifically intended to avoid prediction shift from using a row’s own target or later targets. ([arxiv.org](https://arxiv.org/abs/1706.09516?utm_source=openai)) Recommender offline evaluation also requires a global timeline: learning from interactions unavailable at the prediction point can materially distort apparent accuracy. ([arxiv.org](https://arxiv.org/abs/2010.11060?utm_source=openai))

## When to use / avoid
**Use when:** IDs or metadata are sparse enough that aggregate evidence complements learned embeddings; video creator/category fields provide useful pooling; and the feature service can execute point-in-time joins. These statistics often help distinguish a new or weakly embedded entity through its creator/category history, while user-by-attribute rates provide a low-cost personalization signal.

**Avoid when:** the aggregate table was built from a static full dataset; event ordering is unreliable; labels reflect outcomes without a compatible denominator; or the business objective requires active control of popularity concentration. Popularity is heavily skewed in many recommendation settings, so raw counts can dominate a model unless transformed, clipped, or regularized. ([arxiv.org](https://arxiv.org/pdf/2405.20718?utm_source=openai))

## Requirements and implementation
1. Define the prediction timestamp and eligible denominator. For click prediction, prefer prior **impressions** and clicked impressions; do not use clicks alone as a “CTR” denominator.
2. Sort all logs by event time, then a deterministic tie-breaker. At equal timestamps, either batch-score before batch-update or ensure that only earlier sequence keys are visible.
3. Build train-only state. Fit global priors and any encoder/scaler parameters on the training interval only. For validation/test, initialize state with training history and advance it chronologically using only events that precede each row.
4. Emit, at minimum, for `video_id`, `creator_id`, and `category_id` when present: cumulative exposure count, positive count, smoothed positive rate, `log1p(count)`, and time since first/last interaction. Emit 1-day, 7-day, and 28-day versions only if timestamp density supports them.
5. Emit user affinities such as `(user_id, creator_id)` and `(user_id, category_id)` with the same smoothed rate and support. Back off sparse pairs hierarchically: user–attribute → attribute → global prior. A practical score is \((c_{ua}^+ + \alpha_a\hat p_a)/(c_{ua}+\alpha_a)\).
6. Store the feature timestamp, source-window start/end, denominator definition, and training cutoff alongside every feature artifact. Reject joins where `feature_as_of_time >= label_event_time`.

Operational default: use a global positive-rate prior, `alpha` in 10–100 prior exposures for item-level rates, 25–250 for user–attribute pairs, `log1p` counts, and a cap at the training 99.5th percentile before tree models. Treat these as starting heuristics, not universal research-derived constants. Tune \(\alpha\) on a time-forward validation split, separately for cumulative and recent windows.

## Starting configuration and expected effects
Start with: video cumulative and 7-day smoothed rates; creator/category cumulative rates; `log1p` supports; and user-by-category cumulative affinity. Omit user-by-video rates initially unless repeat consumption is common, because they can act as a memorization feature.

Expect effects to be segment-dependent rather than uniform. GAUC can improve when aggregate evidence ranks candidates better within a user, especially for sparse IDs with shared creator/category evidence. nDCG@5 can improve when these features identify currently strong candidates near the top of a slate. Conversely, gains may vanish or reverse for cold entities, rapidly changing inventory, or datasets where logs are strongly shaped by a prior ranker. Do not claim a fixed lift without a time-forward ablation. Temporal preference patterns are known to be heterogeneous, supporting explicit recency checks rather than assuming static popularity. ([arxiv.org](https://arxiv.org/abs/2104.14200?utm_source=openai))

## Diagnostics and risks
**Leakage signatures:** an unusually large offline jump; validation performance that collapses in a strict chronological replay; features identical across rows that should have different as-of times; or early-validation examples receiving counts from late validation. Audit 20 random rows by reconstructing each numerator and denominator from raw events with timestamps before the row.

**Denominator mismatch:** a click count divided by all platform interactions, rather than prior exposures of the relevant candidate set, measures a mixture of attractiveness and exposure policy. If exposure logs are absent, name the feature `historical_positive_count` or `interaction_rate_proxy`, not CTR.

**Popularity feedback:** a feature can reinforce previously exposed items, worsening long-tail coverage even if GAUC or nDCG@5 rises. Report metrics by item-support decile, creator/category support decile, and new-versus-established inventory. Monitor recommendation-list popularity and concentration in addition to relevance.

**Compute risk:** naive point-in-time group-bys are expensive. Use one chronological pass with keyed counters for cumulative features; maintain deque/ring-buffer or bucketed aggregates for finite windows; materialize features once per split; and bound high-cardinality user–attribute state with TTL, minimum-support thresholds, or top-K recent attributes.

## Cheapest check and clean experiment
**Cheapest train-only check:** choose 100 validation impressions. For each, recompute video and creator counts directly from raw training-plus-earlier-validation events, filtering `event_time < impression_time`. Require exact agreement with the materialized features and verify that every validation row’s feature provenance cutoff precedes its target timestamp.

**Clean single-variable experiment:** hold candidate generation, model seed, training rows, loss, negative sampling, and all non-aggregate features fixed. Compare (A) baseline against (B) baseline plus only cumulative video/creator/category smoothed features. Use the same chronological split and replay protocol. Then add user-by-category affinity in a second ablation, not the first. Report overall GAUC and nDCG@5, confidence intervals or repeated-seed variation where feasible, and segmented results by support decile. A large lift that disappears after strict prefix reconstruction is evidence of leakage, not feature value.

## Related cards and sources
Related cards: `dataset.interaction_log_schema`, `dataset.inventory_and_splits`, `dataset.video_metadata_and_statistics`, `evaluation.within_user_metrics`, `task.experiment_protocol`, `task.leakage_policy`, `features.entity_id_embeddings`, `features.causal_behavior_history_features`.

Primary sources: Prokhorenkova et al., *CatBoost: unbiased boosting with categorical features*, arXiv:1706.09516, on ordered target statistics and leakage-aware construction. ([arxiv.org](https://arxiv.org/abs/1706.09516?utm_source=openai)) Ji et al., *A Critical Study on Data Leakage in Recommender System Offline Evaluation*, arXiv:2010.11060, on global-timeline leakage in recommender evaluation. ([arxiv.org](https://arxiv.org/abs/2010.11060?utm_source=openai))

### Audited web sources

- CatBoost: unbiased boosting with categorical features: <https://arxiv.org/abs/1706.09516?utm_source=openai>
- A Critical Study on Data Leakage in Recommender System Offline Evaluation: <https://arxiv.org/abs/2010.11060?utm_source=openai>
- Popularity-Aware Alignment and Contrast for Mitigating: <https://arxiv.org/pdf/2405.20718?utm_source=openai>
- Learning Heterogeneous Temporal Patterns of User Preference for Timely Recommendation: <https://arxiv.org/abs/2104.14200?utm_source=openai>
