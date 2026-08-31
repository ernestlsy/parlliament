# Embedding-plus-MLP pointwise ranker

## Summary and mechanism
For each displayed impression, map high-cardinality categorical fields—at minimum user ID and video ID—to learned embeddings; embed low-cardinality metadata/context fields; concatenate those vectors with normalized numeric aggregates and fixed-length, causally computed behavioral-history features. Feed the result to a residual multilayer perceptron (MLP) and emit one logit for the long-view label. Train with a pointwise binary loss over logged impressions, then rank the impressions within each evaluation user/request by the predicted logit.

This is a deliberately general baseline: embeddings provide compact representations for sparse categorical inputs, while the MLP can learn nonlinear combinations of IDs, metadata, history summaries, and context. Learned user/item embeddings followed by an MLP are established recommender-model components; industrial ranking systems also commonly separate candidate generation from impression-level ranking. ([arxiv.org](https://arxiv.org/abs/1708.05031?utm_source=openai))

Key assumption: useful candidate-specific signal is already present in the concatenated features. A plain pointwise MLP does not explicitly compare candidates in a slate or attend from each candidate to individual prior events.

## When to use / avoid
**Use** as the first serious neural baseline when all features are available at impression time, candidate lists are short, and you need a low-complexity scorer that can absorb mixed tabular inputs. It is particularly useful for establishing whether IDs, metadata, causal histories, and train-prefix aggregates contain enough signal before adding cross, sequence, or slate modules.

**Avoid or deprioritize** it when diagnostics show that the same history must be interpreted differently for different candidate videos—for example, gains appear only after adding candidate-history similarity or attention. Also deprioritize architecture work if this model already saturates both GAUC and nDCG@5; improve labels, exposure handling, feature validity, or experiment quality first.

## Requirements and implementation
1. Build one row per logged displayed impression, with its long-view target and a stable request/user grouping key for GAUC and nDCG@5.
2. Enforce an event-time feature contract: history uses only events strictly before the impression timestamp; popularity/statistical aggregates use training-prefix data only; do not join future video statistics, future labels, or post-impression outcomes into features.
3. Use separate embedding tables for `user_id`, `video_id`, and each categorical metadata/context field. Reserve explicit OOV/unknown buckets and hash or cap extremely large vocabularies if memory is constrained.
4. Transform numeric inputs with `log1p` where heavy-tailed, then standardize using training-split statistics only. Add missingness indicators when absence is meaningful.
5. Concatenate all feature blocks. Use an MLP with residual blocks: `Linear -> normalization -> ReLU/GELU -> dropout -> Linear`, then add the block input when dimensions match. Finish with a scalar linear head; apply sigmoid only for probability reporting, not before a numerically stable binary-cross-entropy-with-logits loss.
6. Split by the prescribed temporal/user protocol, fit vocabularies and normalizers on train only, and save the exact feature-generation cutoff policy with the model artifact.

DLRM likewise treats categorical features through embeddings and combines them with dense features, while noting that embedding tables can dominate memory and require specialized handling at scale. ([arxiv.org](https://arxiv.org/abs/1906.00091?utm_source=openai))

## Starting configuration and expected effects
These are **practical starting defaults, not universal research findings**:

- ID embedding width: 32–64 for medium-scale data; try 16, 32, 64, and 128. Metadata embeddings: 4–16.
- MLP trunk: `[256, 128, 64]` or `[512, 256, 128]`; use 2–4 residual blocks when the concatenated input is wide.
- Dropout: 0.05–0.20; weight decay: `1e-6`–`1e-4`; AdamW learning rate: `3e-4`–`3e-3` with early stopping on validation GAUC and nDCG@5.
- Use balanced minibatches only if needed for optimization, but evaluate on the natural impression distribution. If reweighting is used, document it because it changes the fitted score scale.

Expected effect: compared with linear or popularity-only scorers, the model often improves GAUC when nonlinear interactions among identity, context, and history matter. nDCG@5 improves only when those score improvements reorder the top of each candidate list correctly; a global pointwise-loss improvement alone does not guarantee that. Do not claim expected gain magnitudes without a matched validation experiment.

## Diagnostics and risks
- **Leakage signature:** implausibly strong offline metrics, especially for recent videos or late timestamps; sharp collapse when features are rebuilt with strict prefix cutoffs. Audit several rows manually, including all source event times.
- **Memorization/cold-start signature:** strong seen-user/seen-video results but weak OOV, new-video, or tail-item slices. Reduce ID width/regularize, strengthen metadata features, and report cold-start slices separately.
- **History insufficiency signature:** GAUC is acceptable but nDCG@5 remains flat, and errors cluster where multiple candidates match different past interests. This motivates a candidate-aware history interaction experiment rather than merely deepening the MLP.
- **Optimization signature:** train loss decreases while validation GAUC/nDCG@5 deteriorate. Check duplicate impressions, label delay, excessive embedding capacity, rare-ID handling, and feature-distribution drift.
- **Compute risk:** embedding tables, optimizer states, and sparse updates can dominate memory even when the MLP is small. Measure embedding-table size, batch latency, and OOV rate before scaling widths. ([arxiv.org](https://arxiv.org/abs/1906.00091?utm_source=openai))

## Cheapest check and clean experiment
**Cheap train-only check:** before fitting the neural model, generate every history and aggregate feature for a small, timestamp-sorted train sample twice: once with the production pipeline and once with a simple per-user/per-video streaming reference implementation that updates state only after each event. Require exact agreement for sampled rows, verify that every feature source time is earlier than its impression time, and report missing/OOV rates. This is inexpensive and catches the highest-risk error: temporal leakage.

**Clean single-variable experiment:** hold splits, labels, optimizer budget, seed set, feature pipeline, and evaluation code fixed. Compare (A) the full embedding-plus-residual-MLP scorer with (B) the identical model after replacing causal behavioral-history features with neutral values plus the same missingness indicators. Report overall and slice-level GAUC and nDCG@5, along with confidence intervals across seeds or bootstrap resamples. This isolates the incremental value of causal history without conflating it with a new interaction architecture.

## Related cards and sources
Related cards: `features.entity_id_embeddings`, `features.causal_behavior_history_features`, `features.train_prefix_smoothed_popularity_features`, `task.leakage_policy`, `evaluation.within_user_metrics`, `task.experiment_protocol`, `dataset.interaction_log_schema`, `dataset.inventory_and_splits`.

Primary sources: Covington, Adams, and Sargin, *Deep Neural Networks for YouTube Recommendations* (RecSys 2016); He et al., *Neural Collaborative Filtering* (arXiv:1708.05031); Naumov et al., *Deep Learning Recommendation Model for Personalization and Recommendation Systems* (arXiv:1906.00091). ([research.google](https://research.google/pubs/deep-neural-networks-for-youtube-recommendations/?utm_source=openai))

### Audited web sources

- Neural Collaborative Filtering: <https://arxiv.org/abs/1708.05031?utm_source=openai>
- Deep Learning Recommendation Model for Personalization and Recommendation Systems: <https://arxiv.org/abs/1906.00091?utm_source=openai>
- Deep Neural Networks for YouTube Recommendations: <https://research.google/pubs/deep-neural-networks-for-youtube-recommendations/?utm_source=openai>
