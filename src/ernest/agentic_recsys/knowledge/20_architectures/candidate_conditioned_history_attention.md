# Candidate-conditioned behavioral-history attention ranker

## Summary and mechanism
For each candidate video, construct a separate user-interest vector by attending over that user’s **causally truncated** recent-video history. This is the Deep Interest Network (DIN) idea: rather than compressing every user into one fixed history embedding, score each historical item for relevance to the current candidate, normalize the scores across valid history positions, and take their weighted sum. The resulting candidate-conditioned interest vector is fused with candidate video-ID and metadata embeddings, user-ID embedding, context/impression-time features, and aggregate features in a final MLP scorer.

A practical local-activation input for history item embedding `h_i` and candidate embedding `v` is `[h_i, v, h_i-v, h_i*v]`, followed by a small MLP producing attention logit `a_i`. Use a masked softmax over available history items and pool `z(v)=sum_i softmax(a)_i h_i`. DIN introduced this candidate-relative pooling to represent diverse user interests differently for different targets. Source: Zhou et al., *Deep Interest Network for Click-Through Rate Prediction*, KDD 2018, DOI: `10.1145/3219819.3219823`; arXiv: `1706.06978`. ([arxiv.org](https://arxiv.org/abs/1706.06978?utm_source=openai))

Assumptions: prior watched/clicked videos contain useful preference signals; candidate and history representations share meaningful semantic space; and timestamps permit a strict as-of impression cutoff. Attention weights are relevance signals, not reliable causal explanations.

## When to use / avoid
**Use** when histories are heterogeneous, candidate sets span multiple topics or formats, and a fixed mean/sum history vector blurs which past behaviors matter for a specific candidate. It is especially suitable when recent video IDs plus category, creator, topic, language, duration bucket, or action-type attributes are present at serving time.

**Avoid or defer** when usable histories are mostly empty or extremely short, when candidate scoring latency is tightly bounded, or when post-impression/post-outcome data cannot be excluded safely. A simpler embedding-MLP ranker can be preferable if history coverage is low or candidate-conditioned computation cannot be amortized over a small rerank set.

## Requirements and implementation
1. Build one training row per impression-candidate with an event-time cutoff equal to the prediction timestamp. Keep only behavior events strictly before that cutoff; apply the same logic in validation, test, and serving.
2. Sort history oldest-to-newest, retain the most recent `L` valid events, and provide a padding mask. Start with watched/clicked videos; add action type, dwell/completion bucket, and item attributes only if they are known as of the behavior timestamp.
3. Embed historical video IDs and available historical attributes, then project their concatenation to the candidate embedding width. Share video-ID embeddings between candidate and matching history IDs unless a measured ablation supports separate tables.
4. Form attention features `[h_i, v, h_i-v, h_i*v]`; use a two-layer activation MLP such as widths `64 -> 32 -> 1`, ReLU or Dice-style activation, and masked softmax. DIN also describes data-adaptive activation and minibatch-aware regularization for large sparse models; treat them as optional extensions rather than prerequisites. ([arxiv.org](https://arxiv.org/abs/1706.06978?utm_source=openai))
5. Concatenate pooled interest `z(v)`, candidate features, user embedding, context, aggregate features, history length, and optional recency features; score with a 2–4 layer MLP. Ensure padding contributes neither embeddings nor attention mass.
6. For serving, encode each user history once per request and batch attention over the candidate rerank set. This design costs roughly proportional to `candidates × L`; cap the rerank candidate count or history length before weakening leakage controls.

## Starting configuration and expected effects
Start with embedding width 32–64, history length `L=50` (try 20, 50, 100, 200), attention MLP `64,32`, scorer MLP `256,128,64`, dropout 0–0.2, and AdamW with learning rate around `1e-3` to `3e-4` after validating optimizer behavior on the baseline. Include log-scaled history length and age/recency buckets; optionally decay attention logits or append time-gap features, but do not assume recency alone replaces relevance matching.

Relative to a matched fixed-pooling history baseline, candidate-conditioned attention can improve GAUC and nDCG@5 when different candidates activate different parts of a user’s history. No improvement magnitude should be presumed: effects depend on history coverage, candidate diversity, label noise, retrieval quality, and whether metadata already captures most relevance. DIN reports superior experimental results against its chosen baselines, but that does not establish a transferable effect size for this video-ranking setting. ([arxiv.org](https://arxiv.org/abs/1706.06978?utm_source=openai))

## Diagnostics and risks
- **Leakage:** sudden offline gains, implausibly strong same-session performance, or degradation after replay usually indicate an as-of join, timestamp, split, or feature-publication error. Audit raw event time, ingestion time, and feature availability separately.
- **Attention collapse:** near-uniform weights suggest weak candidate/history compatibility or excessive regularization; persistent one-item dominance may reflect duplicate events, ID leakage, or a shortcut feature. Log attention entropy, max weight, effective history length, and selected-item ages by cohort.
- **Sparse-history failure:** measure GAUC and nDCG@5 separately for zero, 1–5, 6–20, and >20 usable events. A global average can hide regressions for cold users.
- **Compute failure:** monitor p50/p95 scoring latency and cost against history length and rerank-set size. If latency rises without quality gain, reduce `L`, prefilter histories by recency/type, or restrict the module to the final reranker.
- **Position/exposure confounding:** click or watch labels can encode prior ranker exposure. Keep comparisons on identical logged candidate sets where possible and report within-user ranking metrics alongside aggregate GAUC.

## Cheapest check and clean experiment
**Cheap train-only check:** train a small fixed-pooling baseline and this module on the same training prefix, using an internal time-based holdout drawn only from the training period. Verify: (a) every history event precedes its impression; (b) masked attention sums to one for nonempty histories and zero for empty ones; (c) candidate permutation changes the pooled vector for users with multi-topic histories; and (d) gains are concentrated in sufficiently long-history cohorts rather than caused by malformed rows.

**Clean single-variable experiment:** hold retrieval candidates, labels, temporal split, embeddings, feature set, scorer, optimizer, parameter budget as closely as feasible, and serving truncation constant. Replace only fixed mean/sum history pooling with candidate-conditioned local activation pooling. Evaluate GAUC and nDCG@5 overall and by history-length, candidate-topic-diversity, and activity cohorts; also compare latency and memory. Pre-register the primary metric and rollback threshold before inspecting results.

## Related cards and sources
Related cards: `features.causal_behavior_history_features`, `features.entity_id_embeddings`, `architecture.embedding_mlp_ranker`, `task.leakage_policy`, `evaluation.within_user_metrics`, `task.experiment_protocol`, `dataset.interaction_log_schema`, `dataset.inventory_and_splits`.

Primary source: Guorui Zhou et al. (2018), *Deep Interest Network for Click-Through Rate Prediction*, Proceedings of KDD 2018, pp. 1059–1068, DOI `10.1145/3219819.3219823`, arXiv `1706.06978`. The paper defines DIN’s local activation unit and candidate-specific adaptive interest representation. ([arxiv.org](https://arxiv.org/abs/1706.06978?utm_source=openai))

### Audited web sources

- Deep Interest Network for Click-Through Rate Prediction: <https://arxiv.org/abs/1706.06978?utm_source=openai>
