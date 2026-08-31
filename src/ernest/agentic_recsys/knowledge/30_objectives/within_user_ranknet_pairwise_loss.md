# Within-user RankNet pairwise ranking loss

## Summary and mechanism
For each displayed candidate group belonging to one user (for example, one impression slate or request), form pairs only between a positive long-view example i and a negative example j from that same group. Let the model scores be sᵢ and sⱼ. Optimize the mean logistic pair loss: `softplus(−(sᵢ − sⱼ))`, equivalently `−log sigmoid(sᵢ − sⱼ)`. This is the RankNet pairwise cross-entropy objective: it raises the positive score relative to the negative score rather than independently calibrating each label. RankNet introduced this probabilistic pairwise formulation for learning ranking functions. ([microsoft.com](https://www.microsoft.com/en-us/research/publication/learning-to-rank-using-gradient-descent/?utm_source=openai))

The key assumption is local comparability: a positive and a negative shown in the same user candidate group shared the request-time context, candidate-generation constraints, and exposure opportunity closely enough that their desired order is meaningful. This is more aligned with within-user AUC/GAUC than pairs created across users. Pairwise optimization for implicit feedback is also the central idea of BPR, although this card deliberately restricts negatives to the displayed group instead of treating arbitrary unobserved items as negatives. ([ismll.uni-hildesheim.de](https://www.ismll.uni-hildesheim.de/pub/pdfs/Rendle_et_al2009-Bayesian_Personalized_Ranking.pdf?utm_source=openai))

## When to use / avoid
Use when validation GAUC is the binding metric, impression groups commonly contain both long views and non-long-views, and training batches can preserve group boundaries. It is a practical ranking-focused alternative when pointwise BCE produces reasonable calibration but weak ordering.

Avoid or defer when few groups contain both labels, labels are heavily delayed or censored, group IDs do not represent a real jointly displayed candidate set, or pairing would require cross-user negatives. Do not interpret an unobserved item outside the slate as a negative solely to manufacture pairs.

## Requirements and implementation
Require a binary long-view target and a stable within-user candidate-group ID. Construct pairs after applying the train split, label-finalization window, and all eligibility filters. For group g, collect P₍g₎ = positives and N₍g₎ = negatives; skip groups with either set empty.

1. Sort examples by immutable event ID before sampling.
2. For each positive, sample K negatives from N₍g₎ without replacement when possible; use a stateless seed derived from `(global_seed, epoch, group_id, positive_event_id)`.
3. Compute scores in the normal forward pass and gather the selected positive/negative scores.
4. Minimize the mean pair loss, weighting every group equally first; this prevents very large slates from dominating solely through pair count.
5. Log eligible-group rate, pairs per group, sampled-negative reuse, mean score margin, and loss by group-size bucket.

Never create pairs across a train/validation/test boundary. Features, group membership, candidate-set statistics, and long-view labels must be available at the modeled ranking time; post-impression aggregates and future behavior create leakage. Preserve duplicate-item and repeated-impression policy consistently between objective and evaluation.

## Starting configuration and expected effects
Start with K = 4 negatives per positive, capped at 16 pairs per positive and 128 total pairs per group. Try K in {1, 2, 4, 8, 16}; increasing K raises cost and can overemphasize easy negatives. Sample uniformly within the displayed negatives first. If diagnostics show most sampled margins are already large, test a deterministic mixed sampler: half uniform and half from the model's current top-scoring negatives, refreshed only between epochs.

Use loss weight 1.0 when replacing BCE. If retaining BCE for score calibration or sparse-label stability, start with `L = L_BCE + λ L_pair`, with λ in {0.1, 0.25, 0.5, 1.0}; tune λ on validation GAUC, then inspect nDCG@5 and calibration separately. Expected effects are empirical rather than guaranteed: this objective often improves within-group ordering when valid mixed-label groups are plentiful, so GAUC may improve; nDCG@5 can improve if long-view positives are promoted near the top, but can remain flat or decline if sampled pairs do not represent top-rank errors. Do not claim a lift without a controlled experiment.

## Diagnostics and risks
A low eligible-group rate means the objective has little effective data; report the fraction of groups with at least one positive and one negative, not only the raw pair count. Exploding pair volume, GPU out-of-memory, or epoch-time growth indicates accidental full Cartesian enumeration. A falling pair loss with flat GAUC can indicate duplicated/easy negatives, incorrect group IDs, or a mismatch between training candidates and evaluated candidates.

Watch for position and exposure bias: a negative may mean “not watched under its placement,” not “inferior content.” If positive examples systematically occurred later than negatives, group-level pair loss can reinforce logging-policy artifacts. Also inspect positive-minus-negative margins by position, device, slate size, and user-activity bucket. Large gains only in groups with suspiciously broad timestamps are a data-contract warning.

## Cheapest check and clean experiment
**Cheap train-only check:** after all split and label filters, compute per-group positive and negative counts; report eligible-group rate, median `|P₍g₎|`, median `|N₍g₎|`, and the pair count under K = 4. Then verify that every sampled pair has identical user ID and candidate-group ID, different event IDs, and a positive timestamp no later than the training cutoff. This requires no model training.

**Clean experiment:** hold architecture, features, optimizer, batch size, training steps, split, and deterministic negative-sampling seed fixed. Compare the current pointwise BCE baseline against BCE plus within-user RankNet at λ = 0.25, 0.5, and 1.0 with K = 4. Select only by validation GAUC; report test GAUC, nDCG@5, eligible-group coverage, calibration, runtime, and confidence intervals over fixed resampled evaluation units. After choosing λ, separately vary K; do not tune λ, K, hard-negative mining, and model architecture together.

## Related cards and sources
Related cards: `evaluation.within_user_metrics`, `dataset.interaction_log_schema`, `dataset.inventory_and_splits`, `dataset.population_and_pair_shift`, `dataset.random_exposure_log`, `task.leakage_policy`, `task.experiment_protocol`, `objective.user_normalized_binary_cross_entropy`, `architecture.embedding_mlp_ranker`.

Primary sources: Burges et al., *Learning to Rank using Gradient Descent*, ICML 2005; Microsoft Research Technical Report MSR-TR-2005-06. ([microsoft.com](https://www.microsoft.com/en-us/research/publication/learning-to-rank-using-gradient-descent/?utm_source=openai)) Rendle, Freudenthaler, Gantner, and Schmidt-Thieme, *BPR: Bayesian Personalized Ranking from Implicit Feedback*, UAI 2009. ([ismll.uni-hildesheim.de](https://www.ismll.uni-hildesheim.de/pub/pdfs/Rendle_et_al2009-Bayesian_Personalized_Ranking.pdf?utm_source=openai))

### Audited web sources

- Learning to Rank using Gradient Descent - Microsoft Research: <https://www.microsoft.com/en-us/research/publication/learning-to-rank-using-gradient-descent/?utm_source=openai>
- BPR: Bayesian Personalized Ranking from Implicit FeedbackSteffen Rendle, Christoph Freudenthaler, Zeno Gantner and Lars Schmidt-Thieme: <https://www.ismll.uni-hildesheim.de/pub/pdfs/Rendle_et_al2009-Bayesian_Personalized_Ranking.pdf?utm_source=openai>
