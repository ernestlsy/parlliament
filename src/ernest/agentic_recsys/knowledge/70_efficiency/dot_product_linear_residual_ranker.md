# Dot-product ranker with a small linear residual head

## Summary and mechanism
Use a shared-dimensional user embedding \(u\) and video embedding \(v\) as the primary personalized affinity, then add a cheap residual over impression-time features:

\[
 s(u,v,x)=\frac{u^\top v}{\sqrt d}+b_u+b_v+w^\top \phi(x)
\]

where \(x\) contains only features available at scoring time and \(\phi(x)\) is a normalized numeric / categorical-feature representation. A modest alternative is \(w_2^\top\mathrm{ReLU}(W_1\phi(x)+b_1)\), with one small hidden layer. The dot product carries most user–video variation; the residual learns global or context-dependent corrections such as position, device, hour, freshness, causal popularity, and causal history summaries.

This is an engineering synthesis rather than a separately validated canonical architecture. Its assumptions are that ID embeddings contain substantial collaborative-filtering signal and that remaining useful context effects are mostly additive or shallow. Inner-product user–item models are a central collaborative-filtering baseline, while more flexible neural interaction functions can help when nonlinear interaction structure is genuinely needed. ([hexiangnan.github.io](https://hexiangnan.github.io/papers/www17-ncf.pdf?utm_source=openai)) The two-stage candidate-generation/ranking framing is also established in large-scale video recommendation practice. ([research.google](https://research.google/pubs/deep-neural-networks-for-youtube-recommendations/?utm_source=openai))

## When to use / avoid
**Use** when a stable, low-latency baseline is needed; candidate groups are already available; user and video IDs are well covered; or future retrieval compatibility matters. Cache video embeddings, batch matrix products for a user against candidates, and evaluate the residual only on the resulting candidate rows.

**Avoid** as the final choice when validated gains require candidate-conditioned history attention, rich user–video feature crosses, or strongly nonlinear context interactions. A residual that never changes within a candidate group cannot improve within-group ordering, even if it improves calibration.

## Requirements and implementation
1. Build strictly temporal train/validation/test splits and construct every aggregate, history feature, normalization statistic, and vocabulary from the training prefix appropriate to each impression timestamp.
2. Learn user and video ID embeddings with equal dimension \(d\). Start with separate biases \(b_u,b_v\); regularize biases and embeddings.
3. Include raw or log-transformed numeric context after train-only standardization. Bucket high-cardinality metadata or embed it, but keep the residual small.
4. Use causal history summaries only: for an impression at time \(t\), exclude the target event and every interaction after \(t\). Apply the same rule to popularity, video statistics, and exposure-derived features.
5. Score all candidates belonging to an impression together during evaluation. Do not accidentally replace candidate-specific features with group-level features during joins.

Practical defaults: \(d=32\)–128; L2 regularization around \(10^{-6}\)–\(10^{-4}\) as a sweep, adjusted for loss scale; dropout 0–0.1 in a one-layer residual; 16–64 hidden units if using the nonlinear variant. Start with linear residual features before adding a hidden layer. A useful implementation check is to report the standard deviation of dot-product, bias, and residual terms separately; large residual scale often signals unnormalized inputs or leakage.

## Starting configuration and expected effects
Start with binary cross-entropy, one score per user–video impression, and the linear form above. Train embeddings and residual jointly. If user activity varies substantially, also compare a user-normalized loss as a separate objective experiment rather than silently changing both architecture and weighting.

Expected metric effect is empirical and dataset-specific: relative to an ID-only dot product, causal context and metadata corrections can improve GAUC and nDCG@5 if they alter the ordering of videos within a user’s candidate group. Improvements may be small or absent when these features are group-constant, stale, weak, or redundant with IDs. Compared with a wider MLP ranker, this model will usually trade representational capacity for lower comparator cost and lower variance; do not assume either direction of offline metric change without a controlled experiment. Research on neural collaborative filtering supports testing nonlinear interaction models, but does not establish that they dominate for a particular log, split, or ranking metric. ([hexiangnan.github.io](https://hexiangnan.github.io/papers/www17-ncf.pdf?utm_source=openai))

## Diagnostics and risks
- **Leakage:** suspiciously large validation lifts after adding “recent” statistics, completion summaries, or history features; offline features that use events later than the impression; train/serve feature timestamp mismatch. Audit a few rows manually with source-event timestamps.
- **Residual dominates score:** the residual term has much larger variance than \(u^\top v\), and cold-start or temporal holdout results worsen. Normalize, clip extreme numeric values, increase regularization, or remove leaky/stale features.
- **No nDCG@5 movement but better loss/calibration:** likely the added features are constant within candidate groups or affect all candidates similarly. This can still help probability calibration, but it is not a ranking gain.
- **Poor cold entities:** unseen users/videos default to an unknown embedding and bias. Ensure metadata/context residual features remain available, and report warm/cold slices separately.
- **Compute regression:** per-candidate residual lookup or wide sparse features can erase the dot-product advantage. Precompute video-side metadata embeddings, fuse simple transforms, and profile p50/p99 latency at production-like batch sizes.
- **Popularity shortcut:** causal popularity may raise random-split metrics while weakening temporal or population-shift robustness. Evaluate by recency, item age, user activity, and head/tail popularity.

## Cheapest check and clean experiment
**Cheap train-only check:** fit a ridge/logistic probe on frozen dot-product score, user bias, video bias, and proposed residual features using a train-prefix sub-split. Then inspect (a) feature availability timestamps, (b) residual-feature missingness by split, and (c) within-impression variance. If a feature has near-zero within-group variance, it is unlikely to move nDCG@5 directly; if its apparent predictive power disappears after enforcing as-of timestamps, reject it before full training.

**Clean single-variable experiment:** hold candidate groups, temporal split, labels, optimizer, embedding dimension, parameter budget as far as feasible, and checkpoint rule fixed. Compare: (A) \(u^\top v+b_u+b_v\) versus (B) the same model plus the linear causal residual. Report GAUC and nDCG@5 overall and by warm/cold entity, time period, user activity, and popularity slices, along with latency and score-component variance. Only after this comparison should the linear residual be replaced by one 16–64-unit hidden layer.

## Related cards and sources
Related cards: `features.entity_id_embeddings`, `features.causal_behavior_history_features`, `features.train_prefix_smoothed_popularity_features`, `architecture.embedding_mlp_ranker`, `architecture.candidate_conditioned_history_attention`, `objective.user_normalized_binary_cross_entropy`, `evaluation.within_user_metrics`, `task.leakage_policy`, `evaluation.frozen_candidate_group_integrity_audit`, `evaluation.stratified_temporal_population_evaluation`, `evaluation.probability_calibration_and_ranking_error_audit`.

Primary sources: Covington, Adams, and Sargin, *Deep Neural Networks for YouTube Recommendations* (RecSys 2016), DOI: `10.1145/2959100.2959190`. ([research.google](https://research.google/pubs/deep-neural-networks-for-youtube-recommendations/?utm_source=openai)) Rendle, *Factorization Machines* (ICDM 2010), DOI: `10.1109/ICDM.2010.127`. ([ndlsearch.ndl.go.jp](https://ndlsearch.ndl.go.jp/books/R100000136-I1361981470525523200?utm_source=openai)) He et al., *Neural Collaborative Filtering* (WWW 2017), DOI: `10.1145/3038912.3052569`. ([hexiangnan.github.io](https://hexiangnan.github.io/papers/www17-ncf.pdf?utm_source=openai))

### Audited web sources

- Neural Collaborative Filtering∗: <https://hexiangnan.github.io/papers/www17-ncf.pdf?utm_source=openai>
- Deep Neural Networks for YouTube Recommendations: <https://research.google/pubs/deep-neural-networks-for-youtube-recommendations/?utm_source=openai>
- Factorization Machines | NDLサーチ | 国立国会図書館: <https://ndlsearch.ndl.go.jp/books/R100000136-I1361981470525523200?utm_source=openai>
