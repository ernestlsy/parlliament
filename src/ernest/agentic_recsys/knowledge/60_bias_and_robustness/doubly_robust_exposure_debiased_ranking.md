# Doubly robust exposure-debiased ranking objective

## Summary and mechanism
Train a group-aware ranker with a doubly robust (DR) surrogate: an imputation-model loss for every eligible impression/candidate plus a propensity-weighted residual for labels actually observed. For example, for observed indicator `o`, long-view label `y`, ranker loss `ell`, propensity `p_hat(x)`, and imputed loss `m_hat(x)`, optimize `m_hat(x) + o * (ell(y, s_theta(x)) - m_hat(x)) / max(p_hat(x), p_min)`. The intended DR property is that, before clipping and under the relevant identification assumptions, the target risk can remain correct if either the propensity model or the imputation model is correctly specified; the imputation term can also reduce IPS variance. Wang et al. develop this imputed-error-plus-propensity construction for MNAR recommendation, while Oosterhuis develops DR estimation specifically for position-biased click-feedback ranking. ([proceedings.mlr.press](https://proceedings.mlr.press/v97/wang19n.html?utm_source=openai))

This is not a license to treat logged long views as unconfounded. Require overlap, correctly defined exposure/observation, and nuisance features available at impression time. Clipping is a practical bias-variance intervention, not part of the exact unbiasedness guarantee.

## When to use / avoid
Use when IPS-only optimization is visibly high variance, propensity overlap is acceptable, and there is enough randomized or policy-diverse exposure to train and validate nuisance models. Avoid when propensities and imputations cannot be independently checked, important contexts have near-zero exposure probability, or fold separation cannot be enforced. Deterministic logging or missing candidate inventories make counterfactual claims especially fragile.

## Requirements and implementation
1. Build rows at the displayed-candidate level with candidate-group ID, display position, long-view label/observation flag, propensity, and a temporal or fold ID.
2. Estimate `p_hat` using train-fold data only and pre-impression features; retain known randomized propensities when available.
3. Fit an imputation model for the ranker loss or long-view probability using the same availability restrictions. Do not let it read post-impression watch duration, later engagements, future popularity, or labels from the scored fold.
4. Use K-fold cross-fitting (start with 3–5 folds) or strict temporal nuisance-model isolation: score each fold only with nuisance models not fitted on that fold.
5. Compute DR pointwise losses, then aggregate within displayed candidate groups so high-cardinality users/groups do not silently dominate. Pairwise or listwise extensions must use only valid within-group comparisons and an explicitly defined pair/list observation probability.
6. Train the final scorer on cross-fitted DR losses; select checkpoints on a frozen validation protocol, not on reused nuisance-training labels.

## Starting configuration and expected effects
Start with a binary long-view loss, 3-fold temporal cross-fitting, propensity floor `p_min` chosen from train-only overlap diagnostics, and a small grid such as the 0.5th, 1st, and 2nd percentiles of nonzero estimated propensities. Also report the unclipped-weight tail. Prefer normalized group loss and the same architecture, features, optimizer, and training budget across ordinary, IPS, and DR arms.

Expect DR to help GAUC and nDCG@5 only when the imputation model predicts residual loss well enough to dampen unstable IPS corrections. It may improve stability more reliably than mean ranking metrics. Do not prestate a gain magnitude: clipping, outcome-model misspecification, and limited overlap can instead make DR match or underperform ordinary training or IPS.

## Diagnostics and risks
Log propensity histograms by position, user cohort, item cohort, and candidate-group size; effective sample size of inverse weights; clipped-weight mass; imputation calibration; residual-loss calibration among observed examples; and per-fold nuisance performance. Warning signatures include: a few low-propensity rows dominating gradient norm; large sensitivity to small changes in `p_min`; DR differing sharply by fold; excellent observed-label fit but poor randomized-exposure evaluation; or strong gains only when post-impression features accidentally enter a nuisance model.

Primary risks are leakage, positivity violations, propensity-model error, outcome extrapolation, and extra compute from K nuisance fits. DR does not repair an invalid logging record, omitted candidate set, or unmeasured exposure confounder. Position bias may also require a click/examination model rather than a generic item-observation propensity. ([doi.org](https://doi.org/10.1145/3569453?utm_source=openai))

## Cheapest check and clean experiment
**Cheap train-only check:** split training data temporally; fit propensities and imputations on the prefix; score the suffix. Compare ordinary, clipped IPS, and DR losses on identical rows. Inspect whether DR has lower fold-to-fold loss variance and lower concentration of gradient mass in the largest inverse-weight tail than IPS, without relying on a test set.

**Clean experiment:** predeclare three arms—ordinary, clipped IPS, and cross-fitted DR—with identical ranker, seed set, group-complete batching, feature snapshot, clipping grid, and early-stopping rule. Change only the objective. Evaluate on frozen candidate groups using GAUC and nDCG@5, stratify by exposure propensity and position, and report confidence intervals across seeds and temporal folds. A randomized-exposure holdout is the strongest available adjudicator.

## Related cards and sources
Related IDs: `dataset.interaction_log_schema`, `dataset.inventory_and_splits`, `dataset.random_exposure_log`, `task.leakage_policy`, `evaluation.within_user_metrics`, `evaluation.random_exposure_generalization_audit`, `evaluation.frozen_candidate_group_integrity_audit`, `training.group_complete_stratified_minibatching`, `robustness.clipped_ips_within_group_rank_loss`.

Primary sources: Wang, Zhang, Sun, and Qi, *Doubly Robust Joint Learning for Recommendation on Data Missing Not at Random*, ICML 2019, PMLR 97:6638–6647. ([proceedings.mlr.press](https://proceedings.mlr.press/v97/wang19n.html?utm_source=openai)) Oosterhuis, *Doubly-Robust Estimation for Correcting Position-Bias in Click Feedback for Unbiased Learning to Rank*, ACM TOIS, DOI: `10.1145/3569453`. ([doi.org](https://doi.org/10.1145/3569453?utm_source=openai))

### Audited web sources

- Doubly Robust Joint Learning for Recommendation on Data Missing Not at Random: <https://proceedings.mlr.press/v97/wang19n.html?utm_source=openai>
- Doubly Robust Estimation for Correcting Position Bias in Click Feedback for Unbiased Learning to Rank | ACM Transactions on Information Systems: <https://doi.org/10.1145/3569453?utm_source=openai>
