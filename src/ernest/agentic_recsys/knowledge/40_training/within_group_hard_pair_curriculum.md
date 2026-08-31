# Warm-started hard-pair curriculum within displayed groups

## Summary and mechanism
Train a pairwise ranker only on valid positive–negative item pairs drawn **within the same displayed candidate group** (for example, a user request, feed slate, or impression group). Begin with uniform random valid pairs, then gradually allocate more sampling probability to pairs with small current signed margin \(m=s^+-s^-\) or violations \(m\le0\). This concentrates updates on comparisons that can still alter the within-group order, while a persistent random-pair component preserves coverage.

The mechanism assumes that labels are sufficiently reliable, candidate groups define the ranking context, and current scores are generated without seeing held-out outcomes. Pairwise/LambdaRank-style objectives are appropriate because ranking metrics depend on within-query order; LambdaRank explicitly connects pairwise updates to changes in ranking quality measures. ([proceedings.neurips.cc](https://proceedings.neurips.cc/paper/2006/hash/af44c4c56f385c43f2529f9b1b018f6a-Abstract.html?utm_source=openai)) Hard-example selection is a broadly established optimization idea, but its transfer from detection to this ranking recipe is an engineering adaptation rather than a directly established result for every recommender setting. ([cv-foundation.org](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Shrivastava_Training_Region-Based_Object_CVPR_2016_paper.pdf?utm_source=openai))

## When to use / avoid
Use after a pairwise or LambdaRank-style baseline is already competitive, groups contain many valid positive–negative comparisons, and validation nDCG@5 has plateaued while obvious ordering errors remain. Avoid mining from a near-random model: first obtain a pointwise or uniformly sampled pairwise warm start. Also avoid or heavily constrain it when hard pairs are likely mislabeled, when group sizes yield only one or two pairs, or when the score used for mining leaks future labels.

## Requirements and implementation
Required fields are: `group_id`, long-view label, and the model score. In each training epoch:

1. Construct pairs only where \(y_i>y_j\) and both items belong to the same `group_id`.
2. Warm start for 1–3 epochs, or until train pair accuracy is clearly above chance, using uniformly sampled valid pairs.
3. Score train rows with a model checkpoint that is not updated during the mining pass. Compute \(m_{ij}=s_i-s_j\).
4. Define hardness as `max(0, margin_target - m)` or rank pairs by ascending margin. A simple default is `margin_target=0`; violations rank hardest, followed by near ties.
5. For every group, sample at most `K=8–32` pairs per update contribution. Allocate `r=0.25–0.50` of sampled pairs uniformly at random and `1-r` from the hard-pair distribution. Start with `r=0.50`, then linearly decrease to `0.25` over 2–5 later epochs.
6. Cap a group’s total loss weight, preferably by sampling a fixed number of pairs per group rather than enumerating all \(P_gN_g\) pairs. Use group-complete or group-aware minibatches.
7. Refresh scores and mined pairs every epoch initially; for expensive models, refresh every 2–5 epochs and retain the random component.

Use only train-split labels and train-prefix features when scoring and mining. Never compute hardness from validation/test predictions paired with their labels, and do not use post-impression or future behavior in features available to the mining model.

## Starting configuration and expected effects
A practical initial configuration is: 2 uniform-pair warm-start epochs; then 50% random / 50% hard pairs for 2 epochs; then 25% random / 75% hard pairs; `8–16` pairs per group; and a fresh mining pass each epoch. For very noisy implicit labels, keep at least 40–50% random sampling and exclude extreme suspect cases rather than selecting only the most violated pairs.

Expected effect is empirical, not a guaranteed magnitude: hard-pair emphasis may improve nDCG@5 when the remaining errors are local top-of-list inversions, while GAUC can improve less, remain flat, or deteriorate if the sampler overfocuses on noisy or atypical groups. Compare both metrics, because a top-rank-oriented change can affect them differently. LambdaRank’s motivation is that ranking-quality measures are order-dependent and nonsmooth, not that any particular mining schedule must improve them. ([proceedings.neurips.cc](https://proceedings.neurips.cc/paper/2006/hash/af44c4c56f385c43f2529f9b1b018f6a-Abstract.html?utm_source=openai))

## Diagnostics and risks
**Healthy signature:** the hard subset has lower pair accuracy than random pairs early in mining; violation rate falls across refreshes; validation nDCG@5 improves or stabilizes without a widening train–validation gap.

**Over-mining/noise signature:** training loss continues falling, mined-pair violation rate becomes concentrated in a small set of groups or entities, validation GAUC/nDCG@5 worsens, or selected pairs frequently have ambiguous long-view labels. Raise the random fraction, lower `K`, add a per-group cap, use a near-tie band instead of only maximum violations, or stop the curriculum.

**Compute risk:** exact enumeration is quadratic in positives × negatives per group. Sample a bounded candidate pool per group, score it, then choose hard pairs from that pool. Log group size, candidate-pool size, selected-pair count, margins, and the fraction of updates from the largest 1% of groups.

## Cheapest check and clean experiment
**Cheapest train-only check:** after the warm start, on a fixed train-only audit sample, compare uniformly sampled pairs with candidate mined pairs. Confirm that mined pairs have smaller margins, higher violation rate, and do not come disproportionately from a few groups. This validates that the sampler changes training examples as intended before spending a full experiment.

**Clean experiment:** keep model, data split, optimizer, batch construction, total pair-update budget, and stopping rule fixed. Compare (A) uniform random pairs throughout against (B) the same warm start followed by the curriculum. Run at least several fixed seeds and report GAUC, nDCG@5, per-group coverage, violation rate, and elapsed training cost. The single manipulated variable is pair selection after warm start.

## Related cards and sources
Related cards: `objective.within_user_ranknet_pairwise_loss`, `objective.bce_lambdarank_ndcg5_hybrid`, `training.group_complete_stratified_minibatching`, `evaluation.within_user_metrics`, `task.leakage_policy`, `dataset.inventory_and_splits`, `dataset.population_and_pair_shift`.

Primary sources: Burges, Ragno, and Le, *Learning to Rank with Nonsmooth Cost Functions* (NeurIPS 2006). ([proceedings.neurips.cc](https://proceedings.neurips.cc/paper/2006/hash/af44c4c56f385c43f2529f9b1b018f6a-Abstract.html?utm_source=openai)) Shrivastava, Gupta, and Girshick, *Training Region-Based Object Detectors with Online Hard Example Mining* (CVPR 2016), DOI: `10.1109/CVPR.2016.89`. ([cv-foundation.org](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Shrivastava_Training_Region-Based_Object_CVPR_2016_paper.pdf?utm_source=openai))

### Audited web sources

- Learning to Rank with Nonsmooth Cost Functions: <https://proceedings.neurips.cc/paper/2006/hash/af44c4c56f385c43f2529f9b1b018f6a-Abstract.html?utm_source=openai>
- Training Region-Based Object Detectors With Online Hard Example Mining: <https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Shrivastava_Training_Region-Based_Object_CVPR_2016_paper.pdf?utm_source=openai>
