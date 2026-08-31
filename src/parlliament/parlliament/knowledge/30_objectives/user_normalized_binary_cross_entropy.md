# User-normalized pointwise binary cross-entropy

## Summary and mechanism
Train one logit per labeled impression with binary cross-entropy (BCE), but average BCE within each user candidate group before averaging across groups. For group \(g\), with \(n_g\) scored candidates, label \(y_{gi}\in\{0,1\}\), and model logit \(z_{gi}\), use:

\[
L=\frac{1}{|G|}\sum_{g\in G}\frac{1}{n_g}\sum_{i=1}^{n_g}\bigl(\operatorname{softplus}(z_{gi})-y_{gi}z_{gi}\bigr).
\]

Equivalently, each impression receives weight \(1/n_g\), with a final reduction that gives each group equal total mass. This prevents long candidate lists or prolific users from dominating an objective intended to align with a macro-style within-user GAUC. It remains a pointwise supervised objective: every labeled impression contributes, including all-negative or all-positive groups where pairwise ranking losses have no usable positive-negative pair.

Use logits and a numerically stable BCE-with-logits implementation; do not manually apply sigmoid before the loss. Keep the raw sigmoid probability \(p=\sigma(z)\) in prediction artifacts for calibration and slice diagnostics. Cross-entropy is a probability-estimation loss, whereas GAUC is a within-group ranking metric; direct GAUC-oriented objectives address a different optimization target. ([proceedings.neurips.cc](https://proceedings.neurips.cc/paper/2013/hash/05311655a15b75fab86956663e1819cd-Abstract.html?utm_source=openai))

Assumption: a candidate group is a coherent comparison set, its candidate count is known for every row, and equal group influence is actually desired. If GAUC is exposure- or impression-weighted in evaluation, make the reduction match that definition instead.

## When to use / avoid
**Use when** a dependable first neural-ranker objective is needed; labels exist on every scored training impression; and candidate groups often contain zero, mixed, or all-positive outcomes. It is especially useful as a low-variance reference before adding pairwise or listwise terms.

**Avoid when** labels are absent for a material fraction of scored impressions, when the only success criterion is top-five swapping after this baseline has plateaued, or when inverse candidate-count weighting would contradict the product population to be optimized. For example, equalizing users is inappropriate if business value intentionally scales with impression volume and evaluation does the same.

## Requirements and implementation
1. Build training examples only from the training time window. Attach `group_id`, `candidate_count`, long-view target, features available at scoring time, and a stable row identifier.
2. Compute `candidate_count` from the complete training candidate group, not from a shuffled minibatch or a post-filtered subset. Require \(n_g\ge1\); fail the job for missing, zero, or inconsistent counts.
3. Compute per-row loss with `binary_cross_entropy_with_logits(logit, label, reduction="none")`.
4. Set `weight = 1.0 / candidate_count`, calculate `sum(weight * row_loss) / sum(weight)`, or explicitly mean within groups then mean across groups. The two are equivalent only when the batch represents complete groups; the weighted reduction is usually easier for shuffled batches.
5. Keep loss weighting separate from example eligibility. Do not duplicate positives, mine negatives, or drop displayed zeros without documenting that these changes alter both the target population and calibration interpretation.
6. Log unweighted BCE, weighted BCE, mean candidate count, candidate-count quantiles, total batch weight, positive rate, and gradient norm. Evaluate GAUC and nDCG@5 from the same held-out candidate construction.

Do not derive group counts using future impressions, future inventory, or a user’s eventual session length. That leaks future availability and can make offline ranking look better than a deployable scorer.

## Starting configuration and expected effects
Start with exact inverse-count weighting, no label reweighting, no focal modifier, and standard BCE-with-logits. Use the same optimizer, learning-rate schedule, batch size, features, random seeds, and early-stopping rule as the unweighted BCE baseline.

If candidate counts are extremely heavy-tailed and optimization becomes noisy, treat any denominator cap or softened rule such as \(1/\sqrt{n_g}\) as a separately tested objective, not a harmless implementation detail. A practical empirical sweep is: exact \(1/n_g\), a denominator capped at a high training-count percentile, and \(1/\sqrt{n_g}\). Select only using the prespecified validation metric and population.

Expected direction, not a guaranteed result: equal group weighting commonly makes validation GAUC more representative of typical users when the previous loss was dominated by long lists. nDCG@5 may improve, remain flat, or fall because pointwise BCE does not explicitly emphasize top ranks. Raw-probability calibration may improve for the equal-user target yet worsen for the natural impression-weighted traffic distribution; report both weighted and unweighted calibration diagnostics. These are empirical expectations, not sourced effect sizes. Research on differentiable group-AUC optimization similarly motivates grouping positive-negative comparisons by user because ordinary cross-entropy does not directly optimize AUC/GAUC. ([arxiv.org](https://arxiv.org/abs/2304.09176?utm_source=openai))

## Diagnostics and risks
- **Weighting is inert:** weighted and unweighted losses, gradients, and results are nearly identical despite wide count variation. Check whether counts were accidentally computed per minibatch, cast to integers incorrectly, or normalized away twice.
- **Training instability:** large shifts in total batch weight or gradient norm indicate batches dominated by many tiny groups. Monitor effective group count and use group-aware batching only if needed.
- **GAUC rises but nDCG@5 falls:** the objective is improving broad within-user discrimination without preferentially fixing the top of the slate. Inspect per-rank gain and consider a later ranking-loss comparison rather than silently changing this baseline.
- **Calibration drift:** reliability curves differ sharply between equal-user and impression-weighted aggregation. Preserve untransformed logits and sigmoid probabilities; do not mistake rank-only monotone transformations for calibrated probabilities.
- **Leakage signature:** unusually strong gains occur only when counts or group boundaries are generated after joining labels or future inventory. Rebuild counts from the train-prefix candidate-generation record.
- **Biased feedback:** BCE estimates behavior under logged exposure, not necessarily preference over unexposed items. Randomized-exposure data or an explicit causal treatment is required before interpreting scores as exposure-invariant preference.

## Cheapest check and clean experiment
**Cheap train-only check:** before training, aggregate row weights by group. Every valid group should have total weight approximately one, and the distribution of total group weight should be a point mass at one. Also verify that `candidate_count` equals the number of retained eligible rows for each group; investigate exceptions rather than silently clipping them.

**Clean experiment:** train two otherwise identical runs on identical train/validation/test splits and seeds: (A) ordinary mean BCE and (B) user-normalized BCE. Change no sampler, model, early stopping, candidate construction, or probability post-processing. Report GAUC and nDCG@5 overall and by candidate-count decile, plus unweighted BCE, equal-group BCE, and calibration plots under both impression-weighted and equal-group aggregation. The result is interpretable only if group-count computation and evaluation aggregation are fixed before comparing runs.

## Related cards and sources
Related IDs: `dataset.interaction_log_schema`, `dataset.inventory_and_splits`, `evaluation.within_user_metrics`, `task.experiment_protocol`, `task.leakage_policy`, `task.prediction_artifact`, `dataset.random_exposure_log`, `architecture.embedding_mlp_ranker`.

Primary sources: *On the Relationship Between Binary Classification, Bipartite Ranking, and Binary Class Probability Estimation* (NeurIPS 2013), which analyzes the relationship between probability estimation and ranking; and Sun, Zhang, Zhang, Ren, and Cai, *Enhancing Personalized Ranking With Differentiable Group AUC Optimization*, arXiv:2304.09176, which contrasts cross-entropy with user-grouped GAUC-oriented optimization. ([proceedings.neurips.cc](https://proceedings.neurips.cc/paper/2013/hash/05311655a15b75fab86956663e1819cd-Abstract.html?utm_source=openai))

### Audited web sources

- On the Relationship Between Binary Classification, Bipartite Ranking, and Binary Class Probability Estimation: <https://proceedings.neurips.cc/paper/2013/hash/05311655a15b75fab86956663e1819cd-Abstract.html?utm_source=openai>
- Enhancing Personalized Ranking With Differentiable Group AUC Optimization: <https://arxiv.org/abs/2304.09176?utm_source=openai>
