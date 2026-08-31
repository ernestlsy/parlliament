# Frequency-adaptive user and video embedding dimensions

## Summary and mechanism
Use separate user-ID and video-ID embedding tables whose row width depends on **training-split interaction frequency**: frequent IDs receive higher-capacity vectors and rare IDs receive smaller vectors. Each bucket is projected into a shared scorer width `D`, so downstream feature crosses and the ranking head see a consistent shape:

`e(id) = LayerNorm(E_bucket[id] @ P_bucket)` where `E_bucket[id]` has width `d_bucket` and `P_bucket` maps `d_bucket -> D`.

This is a mixed-dimension embedding design. Its central assumption is that frequently queried entities justify more parameters, while rare entities often cannot reliably estimate wide ID-specific representations. Mixed-dimension embeddings have been studied for recommendation systems specifically, with per-vector dimension scaled by query frequency; the reported motivation and results concern embedding-memory reduction with retained or improved predictive performance. ([arxiv.org](https://arxiv.org/abs/1909.11810?utm_source=openai))

## When to use / avoid
**Use when:** user/video tables and their optimizer states are a material memory cost; train frequencies are strongly long-tailed; and rare IDs show a large train-vs-validation gap or unstable embedding norms.

**Avoid when:** tables are already inexpensive, most IDs have similar support, or bucket boundaries create tiny groups. Do not use validation/test interactions, future logs, impressions, or labels to define frequency buckets: this leaks deployment-time information into capacity allocation.

## Requirements and implementation
1. Freeze a train/validation/test split first. Count user and video occurrences from training interactions only; document whether count means impressions, eligible candidates, or labeled events, and use the same definition for both tables.
2. Bucket users and videos separately by `log1p(count)`. Start with 4 buckets defined by train-count ranges `1`, `2-7`, `8-31`, and `>=32`; replace hand thresholds with train-set quantiles only if every bucket has enough distinct IDs.
3. Assign widths from `{8, 16, 32, 64}` in ascending-frequency order. Set the common projected width `D` equal to the fixed-width baseline, commonly 32 or 64.
4. For every bucket, maintain an embedding table of shape `[num_ids_in_bucket, d_b]` plus a learned projection `[d_b, D]`. Use separate projections for user and video buckets. Keep all non-ID architecture, optimizer, batch schedule, losses, and regularization fixed.
5. Include a reserved OOV/new-ID row in the smallest bucket. At serving time, map IDs absent from the training vocabulary to this row rather than creating a fresh row with an arbitrary width.
6. Compare total **trainable parameters and optimizer-state bytes**, not only embedding-table parameters. Projection matrices and padding/alignment can reduce the apparent saving.

The bucket projection follows the broader variable-capacity embedding pattern: low-frequency symbols can use smaller input representations while preserving compatibility with a common model representation. ([arxiv.org](https://arxiv.org/abs/1809.10853?utm_source=openai))

## Starting configuration and expected effects
A practical empirical default is four buckets with widths `8/16/32/64`, `D=64`, AdamW, and the same weight decay used by the fixed-width baseline. If the tail is exceptionally large, test `4/8/16/32/64`; if it is modest, test `16/32/64` to limit configuration complexity. Constrain widths to multiples of 8 or 16 when that matches accelerator-efficient kernels.

Expected effect: memory and embedding-gradient traffic should decline when many IDs occupy low-width buckets. GAUC and nDCG@5 may improve, remain neutral, or decline; no universal gain magnitude should be assumed. Improvement is plausible when a fixed wide table overfits poorly supported IDs, whereas decline is likely if rare IDs need rich interactions with metadata or if aggressive compression harms tail candidates. The original mixed-dimension work reports that frequency-scaled dimensions can reduce memory substantially while maintaining or improving CTR accuracy, but its benchmark result is not a guarantee for this ranking task or these metrics. ([arxiv.org](https://arxiv.org/abs/1909.11810?utm_source=openai))

## Diagnostics and risks
- **Leakage signature:** results improve unexpectedly after recomputing counts on all data, or bucket membership changes when test data is appended. Fix by deriving and versioning mappings from train data only.
- **Tail under-capacity:** validation GAUC/nDCG@5 falls primarily for users or videos in the smallest buckets; increase only the affected bucket width or merge adjacent low-support buckets.
- **Head over-allocation:** memory reduction is small because high-frequency IDs dominate table bytes; inspect bytes by bucket before widening the tail.
- **Projection bottleneck:** all bucket-specific embeddings collapse to similar projected norms or cosine distributions. Add per-bucket LayerNorm, check projection-gradient norms, and verify that projections are not inadvertently shared across user and video tables.
- **Frequency-shift risk:** a previously rare ID can become common after deployment but remains small until the next vocabulary refresh. Monitor serving-frequency drift and refresh mappings on a defined train-prefix cadence.
- **Misleading speed result:** table memory may fall without faster end-to-end training because sparse lookup, all-to-all communication, and scorer compute can dominate.

## Cheapest check and clean experiment
**Cheap train-only check:** build train-derived frequency histograms for user and video IDs, calculate fixed-table versus mixed-table parameter and optimizer-state bytes, and verify that every validation/test ID maps through the frozen train vocabulary or OOV path. Also report the share of interactions and distinct IDs in each bucket.

**Clean experiment:** train two otherwise identical models with the same seed set and checkpoint rule: (A) fixed user/video widths `D`; (B) frequency-adaptive widths plus projections to `D`. Match the embedding-plus-projection parameter budget as closely as possible, or report both a matched-budget and a matched-width comparison. Evaluate overall and bucket-stratified GAUC and nDCG@5, plus peak memory, step time, and bytes by table. Change no loss, feature, sampling, or split variable in this experiment.

## Related cards and sources
Related card IDs: `dataset.inventory_and_splits`, `task.leakage_policy`, `features.entity_id_embeddings`, `architecture.embedding_mlp_ranker`, `task.experiment_protocol`, `evaluation.within_user_metrics`.

Primary sources: Ginart, Naumov, Mudigere, Yang, and Zou, *Mixed Dimension Embeddings with Application to Memory-Efficient Recommendation Systems*, arXiv:1909.11810. ([arxiv.org](https://arxiv.org/abs/1909.11810?utm_source=openai)) Baevski and Auli, *Adaptive Input Representations for Neural Language Modeling*, arXiv:1809.10853 / ICLR 2019. ([arxiv.org](https://arxiv.org/abs/1809.10853?utm_source=openai))

### Audited web sources

- Mixed Dimension Embeddings with Application to Memory-Efficient Recommendation Systems: <https://arxiv.org/abs/1909.11810?utm_source=openai>
- Adaptive Input Representations for Neural Language Modeling: <https://arxiv.org/abs/1809.10853?utm_source=openai>
