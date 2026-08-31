# Deep & Cross ranker for bounded-order feature interactions

## Summary and mechanism
Start from the embedding-plus-MLP ranker and add a parallel cross branch over the concatenated dense input vector \(x_0\): user-ID and video-ID embeddings, normalized impression-time/context fields, causal history features, and train-prefix aggregates. A standard DCN cross layer is \(x_{l+1}=x_0(w_l^\top x_l)+b_l+x_l\). Each layer raises the maximum explicit interaction degree by one; the deep MLP branch simultaneously learns unconstrained nonlinear transformations. Concatenate the final cross and deep outputs, then score with a small head. This avoids hand-enumerating conjunctions while deliberately biasing the model toward bounded-order field interactions. The original DCN introduced this explicit-cross-plus-deep construction; DCN-V2 increases cross-branch expressiveness through matrix-based and low-rank cross layers. ([arxiv.org](https://arxiv.org/abs/1708.05123?utm_source=openai))

Assume that the fields are aligned to the impression event and that compact tabular interactions—not only ordered behavior dynamics—carry incremental ranking signal. A cross branch can therefore represent patterns such as user-embedding × category, item-embedding × device/hour, or recency × history length, but it cannot repair leakage, missing candidate features, or a behavior representation that discards the relevant sequence information.

## When to use / avoid
Use when an embedding-plus-MLP baseline is already sound, user/item/context/history/aggregate fields coexist at scoring time, and error analysis suggests that affinity changes by category, context, freshness, or prior activity. Evaluate both GAUC and nDCG@5: an improvement is plausible when these interactions affect within-user ordering, but do not assume a particular lift or that both metrics move together.

Avoid or defer when ordered sequences dominate and the current history encoder is weak; first improve the sequence/history representation. Also defer when high-cardinality embeddings are poorly supported, feature coverage is sparse, or validation gains are unstable across time/user slices—extra cross capacity can memorize rare combinations.

## Requirements and implementation
1. Construct \(x_0\) only from features available at the impression timestamp: user and video embeddings; context; causal behavior-history features; and aggregates computed from the training prefix or an equivalently causal online state.
2. Normalize continuous values using training-split statistics; log-transform heavy-tailed counts before normalization. Use explicit missing/unknown indicators where absence is meaningful.
3. Keep the existing MLP branch unchanged for the first comparison. Feed the same \(x_0\) to a cross branch with 2–3 layers, concatenate cross and deep outputs, and use the existing output loss/head.
4. Prefer a low-rank DCN-V2 cross layer when the concatenated vector is wide or a full cross matrix is too costly. Begin with one cross expert; add mixture-of-experts capacity only after a simple cross branch shows repeatable benefit. DCN-V2 was proposed specifically to improve cross-network expressiveness while retaining cost efficiency. ([arxiv.org](https://arxiv.org/abs/2008.13535?utm_source=openai))
5. Preserve identical candidate sets, labels, split boundaries, negative-sampling policy, optimization schedule, and evaluation code relative to the MLP baseline.

## Starting configuration and expected effects
A conservative starting point is: 2 cross layers; one expert; low-rank projection dimension 32–64 for wide inputs; cross-branch dropout 0–0.1; L2/weight decay at least as strong as the baseline; and a 2-layer MLP of 256–128 or the established baseline width. If the input vector is modest, compare the original vector cross layer against low-rank DCN-V2; otherwise start low-rank.

Tune in this order: cross depth {1, 2, 3, 4}; rank {16, 32, 64, 128}; then regularization and learning rate. Stop increasing depth when GAUC/nDCG@5 gains vanish, seed variance rises, or latency/memory no longer fits the ranker budget. The research basis is that DCN explicitly learns bounded-degree crosses and that DCN-V2 offers more expressive cross parameterizations; the exact depth, rank, and metric effect here are practical defaults, not universal findings. ([arxiv.org](https://arxiv.org/abs/1708.05123?utm_source=openai))

## Diagnostics and risks
**Leakage:** verify every history window ends strictly before the impression, popularity/statistical features use only the permitted train prefix or causal event state, and IDs are not encoded through post-outcome aggregates. A large offline gain that disappears on a later temporal holdout is a leakage or shift warning.

**Overfitting signature:** training loss improves sharply, while held-out GAUC is flat/down and nDCG@5 becomes volatile, especially for cold users/items or rare context values. Increase weight decay, reduce rank/depth, cap embedding norms, and inspect performance by frequency bucket.

**Capacity/compute signature:** memory or serving latency grows mainly with the wide concatenated input and cross parameterization. Use low-rank layers, fewer layers, or reduce nonessential embedding widths before changing candidate generation.

**Representation mismatch:** gains appear only for heavily active users while cold-start and sparse-history segments regress. Check whether cross features are dominated by IDs; strengthen metadata/missingness handling and report segmented metrics rather than accepting an aggregate-only win.

## Cheapest check and clean experiment
**Cheapest train-only check:** on the training split, fit a regularized probe containing a small, predeclared set of safe pairwise products—e.g., recency × history length and hour/device × category—alongside the baseline features. Compare held-out loss within a fixed temporal validation slice. If these interactions add no stable signal, prioritize baseline hygiene or sequence features before a larger DCN. This is a screening heuristic, not evidence that the full DCN cannot help.

**Clean experiment:** run an A/B architecture comparison with exactly one changed variable: baseline embedding-plus-MLP versus the same model plus a 2-layer, one-expert low-rank DCN-V2 branch. Use at least three seeds, identical temporal split and candidate lists, and report GAUC plus nDCG@5 overall and by user-history, item-frequency, and time slices. Promote only if the direction is consistent across seeds and the latency/memory change is acceptable; then tune depth and rank separately.

## Related cards and sources
Related cards: `architecture.embedding_mlp_ranker`, `features.entity_id_embeddings`, `features.causal_behavior_history_features`, `features.train_prefix_smoothed_popularity_features`, `task.leakage_policy`, `task.experiment_protocol`, `evaluation.within_user_metrics`, `dataset.interaction_log_schema`.

Primary sources: Wang, Fu, Fu, and Wang, *Deep & Cross Network for Ad Click Predictions* (2017), arXiv:1708.05123. ([arxiv.org](https://arxiv.org/abs/1708.05123?utm_source=openai)) Wang et al., *DCN V2: Improved Deep & Cross Network and Practical Lessons for Web-scale Learning to Rank Systems* (WWW 2021), arXiv:2008.13535; DOI: 10.1145/3442381.3450078. ([arxiv.org](https://arxiv.org/abs/2008.13535?utm_source=openai))

### Audited web sources

- Deep & Cross Network for Ad Click Predictions: <https://arxiv.org/abs/1708.05123?utm_source=openai>
- DCN V2: Improved Deep & Cross Network and Practical Lessons for Web-scale Learning to Rank Systems: <https://arxiv.org/abs/2008.13535?utm_source=openai>
