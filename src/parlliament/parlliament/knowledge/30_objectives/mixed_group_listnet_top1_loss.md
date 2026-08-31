# Mixed-group ListNet top-one loss

## Summary and mechanism
For each intact within-user candidate group \(g\), score every candidate \(s_i\) and form a target top-one distribution from the long-view labels: \(p_i=\operatorname{softmax}(y_i/T_y)\). Form the model distribution \(q_i=\operatorname{softmax}(s_i/T_s)\), with both softmaxes taken **only within that group**. Minimize the mean eligible-group cross-entropy, \(-\sum_i p_i\log q_i\). This is ListNet’s top-one-probability construction: it treats a candidate list, rather than isolated examples or pairs, as the training instance. ([doi.org](https://doi.org/10.1145/1273496.1273513?utm_source=openai))

For binary labels, positive candidates share target mass and negatives receive less mass; the term therefore trains competition at the top of the displayed group. Exclude groups with no positives and groups with all positives: neither supplies the positive-versus-negative within-group contrast targeted here. This exclusion and the optional BCE anchor are practical design choices, not claims established by the original ListNet paper.

## When to use / avoid
Use when displayed candidate groups can be reconstructed exactly, are short enough to score together, and nDCG@5 depends on choosing among several contemporaneous alternatives. Avoid when group IDs are unreliable, candidate sets are truncated differently between train and serving, or most groups are singleton/homogeneous. With sparse binary labels, monitor whether too few mixed groups remain after filtering.

## Requirements and implementation
Require `target_long_view` and a stable within-user candidate-group ID. Split by time/user according to the leakage policy **before** forming batches. Never group candidates across impressions, sessions, requests, or split boundaries; do not use post-ranking outcomes, future behavior, or features computed from the full log.

1. Sort/pack examples by group ID; retain complete groups in one forward pass.
2. Mark a group eligible iff `0 < sum(y) < group_size`.
3. Compute masked segmented softmaxes for labels and scores independently per eligible group.
4. Compute one cross-entropy per group, then average across groups so large groups do not silently dominate.
5. Optionally optimize `L = L_listnet + λ_bce L_BCE`, where BCE uses the same long-view target and is applied per item. Keep loss reductions explicit: group-mean ListNet plus item-mean BCE.

Use numerically stable segmented log-softmax; subtract each group maximum before exponentiating. Assert that target and predicted probabilities each sum to approximately one per eligible group. Preserve all candidates, including negatives, in the group mask.

## Starting configuration and expected effects
Start with `T_y=1`, `T_s=1`, and `λ_bce=0.10`; tune `T_y` over `{0.5, 1, 2}` and `λ_bce` over `{0, 0.03, 0.10, 0.30}`. Lower `T_y` concentrates target mass on higher graded labels; for binary labels it has little useful effect unless labels are transformed into non-binary gains. Clip or regularize scores only if score scale becomes unstable; score temperature and learning rate interact.

Empirically, expect this objective to be most likely to help nDCG@5 when several displayed candidates compete within the same request. GAUC may improve, remain flat, or fall because the loss is normalized within groups and does not directly preserve cross-group score calibration. Do not claim an effect size without a controlled experiment. ListNet was introduced as a listwise method based on permutation/top-k probability models and was evaluated against pairwise baselines in information retrieval; that evidence does not establish effects for this specific long-view recommendation setting. ([doi.org](https://doi.org/10.1145/1273496.1273513?utm_source=openai))

## Diagnostics and risks
Log: eligible-group fraction; group-size distribution; positives per eligible group; target entropy; score entropy; per-group loss; and ListNet/BCE gradient or loss scales. Warning signatures include: near-zero eligible-group coverage (objective is mostly inactive); predicted entropy collapsing early (overconfident winner); uniform predicted distributions (underfitting or masking bug); NaNs after segmented softmax; and improved training list loss with unchanged held-out nDCG@5 (group definition, label noise, or mismatch with evaluation candidates).

The key leakage risk is reconstructing groups using future inventory or outcomes unavailable at ranking time. The key compute risk is padding/packing overhead and accidental cross-group normalization. The key modeling risk is that normalization can learn relative order while discarding useful absolute long-view propensity; the small BCE anchor tests that trade-off.

## Cheapest check and clean experiment
**Cheap train-only check:** on a small deterministic training shard, verify for every eligible group that label-softmax and score-softmax sums are one, gradients are finite, swapping two candidate rows swaps only their corresponding loss contributions, and replacing all scores in a group by the same constant produces predicted uniform probabilities.

**Clean experiment:** hold architecture, features, optimizer, batch candidate groups, seed set, training budget, and early-stopping rule fixed. Compare (A) BCE baseline, (B) ListNet alone, and (C) ListNet plus `λ_bce=0.10`. Evaluate GAUC and nDCG@5 on the identical held-out within-user groups; report eligible-group coverage and confidence intervals across seeds. Only then tune `λ_bce` or temperatures.

## Related cards and sources
Related cards: `objective.user_normalized_binary_cross_entropy`, `objective.within_user_ranknet_pairwise_loss`, `objective.bce_lambdarank_ndcg5_hybrid`, `evaluation.within_user_metrics`, `task.leakage_policy`, `task.experiment_protocol`, `dataset.interaction_log_schema`.

Primary source: Cao, Qin, Liu, Tsai, and Li, *Learning to Rank: From Pairwise Approach to Listwise Approach*, ICML 2007, doi:10.1145/1273496.1273513. ([doi.org](https://doi.org/10.1145/1273496.1273513?utm_source=openai))

### Audited web sources

- Learning to rank | Proceedings of the 24th international conference on Machine learning: <https://doi.org/10.1145/1273496.1273513?utm_source=openai>
