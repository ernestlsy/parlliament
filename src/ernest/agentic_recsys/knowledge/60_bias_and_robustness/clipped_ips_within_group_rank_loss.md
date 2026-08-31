# Clipped IPS-weighted within-group ranking loss

## Summary and mechanism
Observed long-view feedback is conditioned on an item having been displayed, so position or presentation can distort a conventional BCE or pairwise RankNet objective. Inverse-propensity scoring (IPS) reweights observed examples by the inverse probability of their logged exposure under the display policy. Under positivity/overlap and a correctly specified exposure mechanism, this targets risk under a less exposure-skewed target distribution. Counterfactual learning-to-rank work establishes propensity-weighted empirical-risk objectives for biased feedback; counterfactual risk-minimization work also emphasizes that small propensities create high-variance estimates and motivates clipping as an explicit bias–variance trade-off. ([arxiv.org](https://arxiv.org/abs/1608.04468?utm_source=openai))

For impression \(i\), use a stabilized, clipped item weight \(w_i=\min(c,\bar p/p_i)\), where \(p_i\) is the logged or pre-impression-estimated probability that the item receives its realized display/examination opportunity and \(\bar p\) is a fixed reference propensity, such as the training-set mean propensity. Optimize either \(\sum_i w_i\,\mathrm{BCE}(y_i,s_i)\), or only positive–negative pairs within the same complete displayed candidate group \(g\): \(w_{ij}\log(1+\exp(-(s_i-s_j)))\). A practical pair weight is \(w_{ij}=\sqrt{w_iw_j}\), followed by a cap; this aggregation and the operational caps below are empirical engineering choices, not a theorem from the cited work.

Restrict pairs to complete displayed groups: never make a pair across different request, feed, or slate contexts. This preserves the decision set whose relative ordering was actually observed and prevents group composition from being silently turned into a ranking label.

## When to use / avoid
Use when the exposure probability describes the data-generating display mechanism; its inputs are available before impression; and diagnostics indicate adequate overlap between examples receiving different exposure opportunities. Compare against an already strong unweighted BCE or within-group RankNet baseline.

Avoid when propensities are missing, depend on post-impression behavior, or approach zero for a meaningful fraction of training mass. Also avoid treating IPS as a remedy for label-definition error: a long-view threshold that is itself systematically wrong is not fixed by exposure reweighting. Randomized exposure data are especially useful for checking the exposure-bias premise, while jointly learned propensity models require additional identification assumptions. ([arxiv.org](https://arxiv.org/abs/1804.05938?utm_source=openai))

## Requirements and implementation
1. Log one row per displayed candidate with: group identifier, position/presentation fields, long-view label, exposure propensity, and event time. Retain complete candidate groups or mark truncation explicitly.
2. If propensities are not logged, estimate \(P(E=1\mid X)\) using only pre-impression covariates and training-period data. Cross-fit predictions by time fold so an example is not scored by a propensity model trained on its own outcome row. Do not include dwell time, long-view, clicks after display, or features refreshed using future traffic.
3. Choose \(\bar p\) once from the training split. Compute \(w_i=\min(c,\bar p/\max(p_i,\epsilon))\); record the fraction clipped. Normalize weights to mean one within each minibatch, then cap each group’s total normalized weight and renormalize the batch if needed.
4. For weighted RankNet, form pairs only among displayed items in the same group with \(y_i=1,y_j=0\). Sample a bounded number of pairs per group if groups are large, preserving positive–negative coverage. For weighted BCE, apply item weights to all displayed labels but still batch complete groups where feasible.
5. Keep the label, architecture, optimizer, splits, candidate construction, and checkpoint rule identical to the unweighted control.

Compute cost is modest for BCE. Pairwise training can become expensive as group size grows; cap sampled pairs per group and report the realized pair count and effective sample size \((\sum w)^2/\sum w^2\).

## Starting configuration and expected effects
Start with logged propensities if available. Use \(\epsilon=10^{-3}\) only as a numerical floor, not as evidence of overlap. Start clipping at the 99th percentile of raw stabilized weights, then predeclare a small sensitivity grid such as the 95th, 97.5th, and 99th percentiles. Normalize to mean one per minibatch. As an empirical starting guardrail, cap total normalized group weight at 2–4 times the group-size-proportional baseline and sample at most 32–128 positive–negative pairs per group.

The intended effect is reduced dependence of training signal on highly exposed positions or presentations. GAUC and nDCG@5 may improve when exposure bias materially misaligns logged long views with relevance and the propensity model has overlap; they may remain flat or decline when clipping removes too much correction, propensities are misspecified, or the unweighted signal is already well calibrated. Do not promise a magnitude: the cited literature supports the bias/variance rationale, not a universal uplift for these exact metrics or this long-view formulation. ([arxiv.org](https://arxiv.org/abs/1608.04468?utm_source=openai))

## Diagnostics and risks
**Overlap and variance:** plot propensity and raw-weight quantiles by position, label, content cohort, and time. Warning signatures are a heavy weight tail, low effective sample size, one position owning disproportionate weighted loss, or substantial performance sensitivity to small changes in the cap. Clipping reduces variance but introduces bias by construction. ([proceedings.mlr.press](https://proceedings.mlr.press/v37/swaminathan15.html?utm_source=openai))

**Leakage:** propensity estimation must use the serving-time feature snapshot, not outcome-derived or future aggregates. Candidate-group completeness must be established before loss construction; dropped candidates, pagination, or asynchronous logging can invalidate within-group comparisons.

**Objective mismatch:** weighting a negative long-view label presumes its non-long-view outcome is meaningfully observed given its exposure opportunity. If items were merely rendered but not examinable, use an examination/visibility propensity rather than a coarse impression probability.

**Monitoring:** report unweighted and weighted training loss, clipping rate, maximum group contribution, effective sample size, and validation GAUC/nDCG@5 stratified by logged position and propensity decile. A gain only in the weighted objective with deterioration in frozen, ordinary evaluation is not sufficient evidence of a better ranker.

## Cheapest check and clean experiment
**Cheap train-only check:** before training a new ranker, compute raw stabilized weights on the training split. Verify that every propensity decile has nontrivial mass, inspect the top 1% of weights, and compare weighted versus unweighted distributions of pre-impression covariates and positions. If a few examples or groups dominate weighted mass, fail the gate rather than searching repeatedly for a favorable clipping threshold.

**Clean experiment:** train exactly two models from the same seeds and complete-group minibatches: (A) the established unweighted BCE or within-group RankNet objective and (B) the same objective with predeclared stabilized clipping, batch normalization, and group cap. Freeze propensity estimation, candidate groups, feature snapshots, training budget, and checkpoint selection. Evaluate both on the same untouched temporal test set using GAUC and nDCG@5, plus position- and propensity-stratified slices. Run the predeclared clipping sensitivity grid only if (B) passes the overlap gate; otherwise retain (A).

## Related cards and sources
Related cards: `dataset.interaction_log_schema`, `dataset.random_exposure_log`, `dataset.inventory_and_splits`, `task.leakage_policy`, `objective.user_normalized_binary_cross_entropy`, `objective.within_user_ranknet_pairwise_loss`, `training.group_complete_stratified_minibatching`, `evaluation.frozen_candidate_group_integrity_audit`, `evaluation.stratified_temporal_population_evaluation`, `evaluation.random_exposure_generalization_audit`, `evaluation.probability_calibration_and_ranking_error_audit`.

Primary sources: Joachims, Swaminathan, and Schnabel, *Unbiased Learning-to-Rank with Biased Feedback* (2017); Swaminathan and Joachims, *Counterfactual Risk Minimization: Learning from Logged Bandit Feedback* (ICML 2015); Ai et al., *Unbiased Learning to Rank with Unbiased Propensity Estimation* (CIKM 2018). ([arxiv.org](https://arxiv.org/abs/1608.04468?utm_source=openai))

### Audited web sources

- Unbiased Learning-to-Rank with Biased Feedback: <https://arxiv.org/abs/1608.04468?utm_source=openai>
- Unbiased Learning to Rank with Unbiased Propensity Estimation: <https://arxiv.org/abs/1804.05938?utm_source=openai>
- Counterfactual Risk Minimization: Learning from Logged Bandit Feedback: <https://proceedings.mlr.press/v37/swaminathan15.html?utm_source=openai>
