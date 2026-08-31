# Shared-trunk auxiliary-behavior multitask loss

## Summary and mechanism
Use one feature/representation trunk \(h_\theta(x)\) and a separate sigmoid head for the primary long-view label plus one lightweight sigmoid head per auxiliary behavior. Train on impression rows with a weighted sum of *within-user normalized* binary cross-entropy (BCE):

\[
\mathcal L=\mathcal L_{long}+\sum_{a\in A}\lambda_a\mathcal L_a,\qquad
\mathcal L_t=\frac{1}{|U_B|}\sum_{u\in U_B}\frac{1}{|I_u|}\sum_{i\in I_u}\operatorname{BCE}(y_{i,t},\sigma(w_t^\top h_\theta(x_i)+b_t)).
\]

Score candidates with the dedicated long-view head only; auxiliary heads shape the shared representation during training and are not substituted for the serving score. The assumption is that auxiliary behaviors are measured on the same impression universe, are available under the same label-cutoff policy, and share predictive structure with long viewing. Shared representations can help sparse tasks, but task relationships matter: negative transfer is possible when tasks conflict. ([arxiv.org](https://arxiv.org/abs/1804.07931?utm_source=openai))

## When to use / avoid
**Use** when long-view positives are sparse, auxiliary labels are more prevalent, and all labels are valid at impression time after applying a fixed outcome window. Favor earlier or related engagement signals—for example, a qualified play, completion threshold, or explicit positive action—only after checking that they improve the held-out primary metric.

**Avoid** an auxiliary task if it is derived from future information, is missing or differently defined in validation, has very low support, or represents a competing preference. Do not treat a post-impression behavior as an input feature for that same impression. If primary GAUC or nDCG@5 declines in a fixed ablation, remove or sharply down-weight the task rather than assuming more training will repair it.

## Requirements and implementation
1. Build one row per exposed candidate with user/group ID, candidate features, long-view target, and auxiliary labels. Define label windows and censoring rules before splitting data.
2. Use the same causal feature snapshot for every head. Mask labels that are not yet observable at the split cutoff; do not relabel unknown outcomes as negatives without an explicit maturity rule.
3. Reuse the existing ranker trunk. Add linear or small one-layer heads first; auxiliary-head capacity should not dominate trunk capacity.
4. Compute BCE per user candidate group, then average users, so heavy-activity users do not dominate the objective merely through more logged candidates.
5. Apply the long-view head alone for ranking and report GAUC and nDCG@5 from that score. Log per-head prevalence, BCE, AUC where meaningful, gradient norms into the shared trunk, and metrics by user activity bucket.

This is a simple shared-bottom multitask design. More elaborate task-routing architectures exist when task relatedness is weak or heterogeneous, but they add parameters and experiment surface area; start with the shared trunk because the intended intervention is auxiliary supervision rather than architecture replacement. ([doi.org](https://doi.org/10.1145/3219819.3220007?utm_source=openai))

## Starting configuration and expected effects
Start with \(\lambda_{long}=1\). For one auxiliary label, try \(\lambda_a\in\{0.05,0.1,0.2,0.4\}\); for multiple labels, start with total auxiliary weight at 0.1–0.3 and keep no individual task above 0.2 until validated. Use the baseline optimizer, batch construction, and early-stopping rule. Tune weights on a validation period without changing features, model width, negatives, or ranking loss simultaneously.

Expected effect is empirical, not guaranteed: a useful, sufficiently common auxiliary target can reduce variance in shared representations and improve primary GAUC and/or nDCG@5, especially for sparse long-view supervision. Gains need not be monotonic with auxiliary weight, and a task may improve its own BCE while harming long-view ranking. ESMM provides a related example of exploiting sequential behavioral supervision and representation transfer to address sparse downstream outcomes, though its probabilistic structure is not identical to this card. ([arxiv.org](https://arxiv.org/abs/1804.07931?utm_source=openai))

## Diagnostics and risks
- **Leakage signature:** unusually large offline lift, especially near the label-window boundary; feature-importance spikes for fields populated after exposure. Audit event timestamps and availability, then retrain with a strict pre-impression snapshot.
- **Negative transfer:** auxiliary BCE falls while long-view GAUC/nDCG@5 falls; trunk-gradient cosine similarity is persistently negative or one auxiliary gradient norm dominates. Reduce \(\lambda_a\), drop that task, or consider task-specific layers/routing only after the simple ablation fails.
- **Prevalence or maturity problem:** volatile auxiliary metrics across time or cohorts, many labels unresolved at validation, or strong performance only on mature rows. Increase the label delay, mask immature labels, and re-evaluate chronologically.
- **Objective/metric mismatch:** GAUC rises but nDCG@5 does not, suggesting improved global discrimination without better top-of-list ordering. Keep this loss as representation regularization and test a separate primary ranking-objective card later; do not change both interventions in one experiment.
- **Compute risk:** extra heads are usually cheap, but more labels increase I/O, label joins, logging, and backward-pass cost. Measure examples/sec and accelerator memory against the single-task baseline.

## Cheapest check and clean experiment
**Cheap train-only check:** before any full sweep, calculate per-label prevalence, per-user label coverage, overlap with long views, and label maturity after the planned cutoff. Train one short, fixed-budget run with a single auxiliary head at \(\lambda=0.1\). Require finite losses, comparable throughput, non-dominant auxiliary trunk gradients, and no timestamp-policy violation. This check is diagnostic only; do not select a task from training metrics.

**Clean experiment:** hold split, candidate groups, features, trunk, seed set, optimizer schedule, training steps, and primary scorer fixed. Compare: (A) primary long-view BCE only versus (B) A plus exactly one auxiliary BCE at \(\lambda=0.1\). Select using primary validation GAUC and nDCG@5 from the long-view head, then repeat the winning configuration across seeds and at least one later chronological validation slice. Only then tune \(\lambda\); add a second auxiliary label in a new ablation.

## Related cards and sources
Related cards: `dataset.interaction_log_schema`, `dataset.inventory_and_splits`, `evaluation.within_user_metrics`, `task.leakage_policy`, `architecture.embedding_mlp_ranker`, `objective.user_normalized_binary_cross_entropy`, `objective.within_user_ranknet_pairwise_loss`, `objective.bce_lambdarank_ndcg5_hybrid`.

Primary sources: Ma et al., *Entire Space Multi-Task Model: An Effective Approach for Estimating Post-Click Conversion Rate* (arXiv:1804.07931); Ma et al., *Modeling Task Relationships in Multi-task Learning with Multi-gate Mixture-of-Experts*, KDD 2018, DOI: 10.1145/3219819.3220007. ([arxiv.org](https://arxiv.org/abs/1804.07931?utm_source=openai))

### Audited web sources

- Entire Space Multi-Task Model: An Effective Approach for Estimating Post-Click Conversion Rate: <https://arxiv.org/abs/1804.07931?utm_source=openai>
- Modeling Task Relationships in Multi-task Learning with Multi-gate Mixture-of-Experts | Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining: <https://doi.org/10.1145/3219819.3220007?utm_source=openai>
