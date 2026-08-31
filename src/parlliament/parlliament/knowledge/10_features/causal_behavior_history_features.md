# Causally truncated behavioral-history features

## Summary and mechanism
For every scored impression at time `t`, build user-history features from **only events strictly earlier than `t`** under a deterministic per-user order. Typical features are: the last `K` interacted video IDs, time since the most recent event, log-scaled prior-event count, and distributions over categories/tags attached to prior videos. Join candidate-item metadata separately, then let the ranker learn matches such as candidate category versus recent-history category share.

This is a cheap, explicit approximation to sequential preference modeling: recent IDs preserve short-term intent; recency and history length encode activity state; aggregated attributes provide lower-cardinality signals that generalize beyond individual IDs. The causal cutoff is non-negotiable: feature availability at training, validation, test, and serving must match. Sequential-recommendation research models prior behaviors for prediction, while also warning that allowing a representation to see its target/future context creates information leakage. ([arxiv.org](https://arxiv.org/pdf/1904.06690?utm_source=openai))

Assumptions: timestamps or event sequence numbers provide a reliable order; the event type used in history would be observable online; and item attributes are versioned or otherwise available as of scoring time. If same-time ordering is unresolved, exclude ties from one another rather than relying on arbitrary row order.

## When to use / avoid
**Use when:** ordered per-user logs and online-accessible previous interactions exist; you need recency sensitivity without deploying an RNN/Transformer; or a baseline ranker needs a stronger personalization signal. History-conditioned representations are established components of industrial recommendation and CTR models. ([doi.org](https://doi.org/10.1145/2959100.2959190?utm_source=openai))

**Avoid when:** event times cannot be safely ordered; a feature job would read the complete validation/test table; item taxonomy was created after the evaluation period; or history is not available with the same latency and retention limits in production. For anonymous, extremely short sessions, prefer session-level features keyed by session rather than fabricating a durable user history.

## Requirements and implementation
1. Canonically sort each user’s events by `(event_timestamp, stable_event_order, immutable_event_id)`. Define a documented tie policy. The feature row for event `j` may read only rows `< j` for that user.
2. Process each user in one forward pass. Maintain a bounded deque of the most recent `K` eligible video IDs, last-event time, eligible-event count, and sparse category/tag counters. Emit features **before** updating state with the current event.
3. Start with an eligibility policy such as watched/clicked/liked events only; do not silently mix impressions, skips, and conversions. Keep separate counters or feature namespaces if multiple event types are retained.
4. Recommended feature set:
   - `hist_video_id_1...K`, newest first, with PAD and OOV tokens;
   - `seconds_since_last_event`, clipped then transformed as `log1p(seconds)`;
   - `log1p(history_length)` and bucketed history length;
   - decayed event count, e.g. `sum(exp(-age/tau))`;
   - normalized category/tag shares from history, optionally for the top `M` vocabulary values plus OTHER;
   - candidate-aligned features: prior count/share for the candidate’s category or tags, and time since the last history event sharing that attribute.
5. Fit all vocabularies, frequency thresholds, clipping caps, and normalization constants on training data only. Use an as-of item-metadata snapshot if attributes can change.
6. In offline validation/test, initialize state from events strictly before the split boundary, then advance chronologically only through events that would have occurred before each scored row. Do not preload a user’s entire validation history.
7. At serving, use the same state update semantics, including late-arriving-event policy, event deduplication, TTL, and attribute lookup. Log a feature timestamp or source-event watermark for audits.

## Starting configuration and expected effects
Start with `K=20` recent video IDs; test `K ∈ {5, 10, 20, 50, 100}`. Use a recency clip of 30 days for general video feeds, while retaining a missing/no-prior-event indicator. Start exponential-decay half-lives of 1 hour, 1 day, and 7 days; choose according to interaction cadence rather than assuming one global timescale. Cap category vocabulary at roughly 100–1,000 frequent values and tags at roughly 1,000–10,000, depending on data volume; map the rest to OTHER. Apply a minimum support threshold before exposing an attribute feature.

Expected effect is empirical and data-dependent: these features often improve GAUC when users have repeatable preferences and improve nDCG@5 when recent behavior reveals immediate intent. Gains may be small or absent for sparse histories, rapidly changing inventories, weak item metadata, or objectives driven mostly by instantaneous context. Recent-history conditioning is a practical alternative to richer sequential architectures, not a guarantee that it will outperform them. Candidate-specific attention over behavior histories is a more expressive next step when simple aggregates plateau. ([arxiv.org](https://arxiv.org/abs/1706.06978?utm_source=openai))

## Diagnostics and risks
- **Future leakage:** suspiciously large offline lifts, especially strongest near the split boundary; recompute with a train-only history cutoff and compare.
- **Same-row leakage:** a current positive event accidentally appears as `hist_video_id_1`; assert that every source event has order strictly less than the scored row.
- **Validation contamination:** user histories or aggregate counters were built on concatenated train/validation/test logs. Check feature provenance and per-row maximum source timestamp.
- **Serving skew:** offline history lengths or freshness are much larger than online values. Monitor null/PAD rate, history-length distribution, recency quantiles, OOV rate, and source-to-score latency by split and production day.
- **Popularity proxying:** broad category shares can mostly encode user activity or globally popular inventory. Compare against activity-only features and candidate-category popularity controls.
- **Compute/state growth:** unbounded tag maps and full histories cause state bloat. Bound deques, prune low-weight decayed counters, hash or cap long-tail attributes, and periodically compact state.
- **Ordering ambiguity:** many equal timestamps or late events produce unstable features. Measure tie rate and late-arrival rate; if material, use an ingestion sequence number or conservative exclusion policy.

## Cheapest check and clean experiment
**Cheap train-only check:** select a training-time cutoff `T`. Construct features for rows before `T` using only earlier training rows; then reconstruct the same rows from a streaming per-user pass and require exact agreement for sampled users. Assert `(max_source_time < score_time)` for every nonempty feature, and verify the current video ID is absent from its own prior-ID sequence. This catches most join, sort, and update-order mistakes before model training.

**Clean single-variable experiment:** hold candidate generation, labels, split dates, model architecture, training seed set, and all non-history features fixed. Compare: (A) no behavioral-history features versus (B) causal history features with `K=20`, one selected decay half-life, and fixed vocabularies. Evaluate GAUC and nDCG@5 overall and by cold (`0` prior events), light (`1–4`), and established (`5+`) histories. Report feature coverage and online-feasibility checks alongside metrics. Only after this comparison should `K`, decay, event eligibility, and attribute vocabulary be tuned.

## Related cards and sources
Related cards: `dataset.interaction_log_schema`, `dataset.inventory_and_splits`, `dataset.video_metadata_and_statistics`, `evaluation.within_user_metrics`, `task.experiment_protocol`, `task.leakage_policy`, `features.entity_id_embeddings`.

Primary sources:
- Covington, Adams, and Sargin (2016), *Deep Neural Networks for YouTube Recommendations*, RecSys ’16. DOI: `10.1145/2959100.2959190`. ([doi.org](https://doi.org/10.1145/2959100.2959190?utm_source=openai))
- Zhou et al. (2018), *Deep Interest Network for Click-Through Rate Prediction*. arXiv: `1706.06978`. ([arxiv.org](https://arxiv.org/abs/1706.06978?utm_source=openai))
- Hidasi et al. (2016), *Session-based Recommendations with Recurrent Neural Networks*. arXiv: `1511.06939`. ([arxiv.org](https://arxiv.org/abs/1511.06939?utm_source=openai))
- Sun et al. (2019), *BERT4Rec: Sequential Recommendation with Bidirectional Encoder Representations from Transformer*. arXiv: `1904.06690`. ([arxiv.org](https://arxiv.org/pdf/1904.06690?utm_source=openai))

### Audited web sources

- BERT4Rec: Sequential Recommendation with Bidirectional: <https://arxiv.org/pdf/1904.06690?utm_source=openai>
- Deep Neural Networks for YouTube Recommendations | Proceedings of the 10th ACM Conference on Recommender Systems: <https://doi.org/10.1145/2959100.2959190?utm_source=openai>
- Deep Interest Network for Click-Through Rate Prediction: <https://arxiv.org/abs/1706.06978?utm_source=openai>
- Session-based Recommendations with Recurrent Neural Networks: <https://arxiv.org/abs/1511.06939?utm_source=openai>
