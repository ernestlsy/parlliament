# Probability calibration and ranking-error audit

## Summary and mechanism
Ranking metrics answer whether positives are ordered above negatives within a displayed candidate group; calibration asks whether a score of, for example, 0.20 corresponds to an event rate near 20%. Audit both. Report raw-score and post-hoc-calibrated Brier score, binary log loss, reliability bins, and the mean predicted-positive rate against the observed prevalence, overall and by candidate-group stratum. Brier score and log loss are proper probabilistic scores, so they assess probability quality rather than rank ordering. ([rmets.onlinelibrary.wiley.com](https://rmets.onlinelibrary.wiley.com/doi/abs/10.1002/qj.456?utm_source=openai))

A single strictly monotone calibration map preserves within-group ordering, so GAUC and nDCG@5 should ordinarily be unchanged when evaluated from that one calibrated head. Do not expect a rank gain. Differences can arise from ties, clipping, non-monotone procedures, group-specific maps, score blending, or changed downstream selection. Calibration matters even when ranking is stable because thresholds, multitask weighting, and prediction averaging treat score magnitudes as meaningful probabilities.

## When to use / avoid
Use for BCE-trained or binary multitask heads; before averaging heads or checkpoints; and after a ranking gain accompanied by saturated near-0/near-1 scores, collapsed score spread, or implausible predicted prevalence. Avoid using Brier/log loss in place of GAUC or nDCG@5 for rank-model selection. Never fit a calibrator on validation labels and report that same split as its performance. Do not apply a monotone map merely to claim a ranking improvement.

## Requirements and implementation
Persist one canonical validation prediction artifact containing row ID, split/time marker, binary long-view label, raw logit and sigmoid probability, displayed candidate-group ID, and relevant candidate-group attributes. Freeze candidate membership before all calculations.

1. Create a calibration-development partition from training data, respecting the production temporal split. Prefer a dedicated tail-of-training calibration fold; if data are scarce, obtain out-of-fold predictions with 5 folds.
2. Fit calibration only on predictions not produced by a model trained on those rows. Start with logistic/Platt scaling on logits. It is low-variance and easy to deploy. Use isotonic regression only with ample calibration rows and visibly non-sigmoidal distortion; it can overfit small calibration sets. ([mlanthology.org](https://mlanthology.org/icml/2005/niculescumizil2005icml-predicting/?utm_source=openai))
3. Apply the frozen map once to the untouched validation/test predictions. Compute Brier score, log loss, prevalence error `mean(p)-mean(y)`, and reliability tables for raw and calibrated probabilities.
4. Slice every output by displayed candidate-group type, group size bucket, temporal cohort, and score decile. Also aggregate at candidate-group level so large groups do not silently dominate a row-level diagnostic.

Empirical starting defaults: use 10 equal-frequency reliability bins; merge bins with fewer than 100 positive labels or fewer than 500 total rows; clip only for log-loss numerics, e.g. `[1e-6, 1-1e-6]`; and report bin count and clipping explicitly. These are stability-oriented operational defaults, not universal research thresholds.

## Starting configuration and expected effects
Start with one global logistic calibration map per binary head. Compare it with identity/no calibration and, only if the calibration sample is large enough, isotonic regression. Beta calibration is a reasonable third candidate when the reliability curve is asymmetric or logistic calibration worsens probability scores; it was proposed to include the identity map and address distortions that a logistic family may miss. ([proceedings.mlr.press](https://proceedings.mlr.press/v54/kull17a.html?utm_source=openai))

Expected effect: a successful map lowers held-out Brier and log loss and moves reliability-bin event rates toward predicted probabilities. GAUC and nDCG@5 should remain effectively unchanged for a global monotone map; do not fabricate or target a fixed improvement magnitude. If calibrated probabilities are used in an ensemble or threshold rule, evaluate the full downstream rule separately because magnitude changes can alter outcomes.

## Diagnostics and risks
**Overconfidence:** bins at high predicted probability realize much less often; log loss is often especially sensitive to wrong extreme predictions. **Underconfidence:** predictions occupy a narrow middle range while observed rates vary substantially. **Prevalence mismatch:** `mean(p)` differs materially from `mean(y)`, often signaling intercept shift, label-window mismatch, exposure/population shift, or leakage in features or labels. **Mixed-group failure:** acceptable global calibration but opposing errors across candidate-group types; this can corrupt cross-head averaging even while GAUC looks healthy.

Primary risks are leakage and misleading aggregation. Fitting on the evaluated labels makes reliability and proper-score improvements optimistic. Reusing in-sample training predictions also gives an overly favorable calibration fit. Isotonic maps may form step functions and create ties, which can perturb ranking metrics. Group-specific maps may improve a slice while changing cross-group comparability; require a real deployment justification and evaluate each group plus the pooled population. Calibration cannot repair missing features, corrupted labels, or candidate-set changes.

## Cheapest check and clean experiment
**Cheapest train-only check:** generate 5-fold out-of-fold training predictions, fit logistic calibration within each fold using the other folds, then concatenate held-out calibrated predictions. Produce raw-versus-calibrated Brier/log loss, 10-bin reliability tables, prevalence error, and the same tables split by candidate-group type. This requires no retraining of the base ranker.

**Clean single-variable experiment:** hold the trained ranker, candidate sets, labels, evaluation code, and checkpoint fixed. Compare only (A) identity mapping, (B) global logistic calibration fitted on a disjoint train-derived calibration partition, and optionally (C) isotonic calibration fitted on that identical partition. Evaluate once on untouched validation/test data. Report GAUC and nDCG@5 from raw and mapped scores, plus probabilistic metrics and slices; investigate any rank difference rather than interpreting it automatically as improvement.

## Related cards and sources
Related IDs: `task.prediction_artifact`, `task.leakage_policy`, `evaluation.within_user_metrics`, `evaluation.frozen_candidate_group_integrity_audit`, `evaluation.stratified_temporal_population_evaluation`, `objective.user_normalized_binary_cross_entropy`, `objective.shared_trunk_auxiliary_behavior_multitask_loss`, `training.metric_aligned_checkpoint_averaging`.

Primary sources: Zadrozny & Elkan, *Obtaining Calibrated Probability Estimates from Decision Trees and Naive Bayesian Classifiers* (ICML 2001). ([mlanthology.org](https://mlanthology.org/icml/2001/zadrozny2001icml-obtaining/?utm_source=openai)) Niculescu-Mizil & Caruana, *Predicting Good Probabilities with Supervised Learning* (ICML 2005), DOI `10.1145/1102351.1102430`. ([mlanthology.org](https://mlanthology.org/icml/2005/niculescumizil2005icml-predicting/?utm_source=openai)) Kull, Silva Filho & Flach, *Beta Calibration* (AISTATS 2017). ([proceedings.mlr.press](https://proceedings.mlr.press/v54/kull17a.html?utm_source=openai))

### Audited web sources

- Reliability, sufficiency, and the decomposition of proper scores - Bröcker - 2009 - Quarterly Journal of the Royal Meteorological Society - Wiley Online Library: <https://rmets.onlinelibrary.wiley.com/doi/abs/10.1002/qj.456?utm_source=openai>
- Predicting Good Probabilities with Supervised Learning | ML Anthology: <https://mlanthology.org/icml/2005/niculescumizil2005icml-predicting/?utm_source=openai>
- Beta calibration: a well-founded and easily implemented improvement on logistic calibration for binary classifiers: <https://proceedings.mlr.press/v54/kull17a.html?utm_source=openai>
- Obtaining Calibrated Probability Estimates from Decision Trees and Naive Bayesian Classifiers | ML Anthology: <https://mlanthology.org/icml/2001/zadrozny2001icml-obtaining/?utm_source=openai>
