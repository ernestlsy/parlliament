# Predeclared group-DRO within-user ranking objective

## Summary and mechanism
Use **group distributionally robust optimization (group DRO)** when complete displayed candidate groups show repeatable losses in predeclared slices despite acceptable aggregate GAUC or nDCG@5. Assign every displayed candidate group exactly one training-only group label, such as a training-time period bucket, user-history-depth bucket, candidate-group-size bucket, or item-frequency bucket. Then optimize the loss of the currently hardest group rather than only the average loss.

For group losses \(L_g(\theta)\), the core objective is \(\min_\theta \max_{q\in\Delta^G}\sum_g q_g L_g(\theta)\). In practice, maintain adversarial group weights \(q_g\), increase the weight of groups with high moving loss, and train on the weighted mixture. Sagawa et al. describe stochastic group-DRO optimization and show that naïve worst-group training in overparameterized neural networks can fail to improve worst-group generalization unless it is paired with stronger regularization or early stopping. ([mlanthology.org](https://mlanthology.org/iclr/2020/sagawa2020iclr-distributionally/?utm_source=openai))

For ranking, define \(L_g\) as a **user-normalized loss over complete displayed candidate groups**, not as an impression-weighted loss. A practical anchored objective is:

\[
L(\theta)=(1-\alpha)L_{\mathrm{anchor}}(\theta)+\alpha\sum_{g=1}^{G}q_gL_g(\theta)+\lambda\lVert\theta\rVert_2^2.
\]

Use ordinary user-normalized BCE, RankNet, or a BCE-plus-ranking hybrid for \(L_{\mathrm{anchor}}\); compute the same base loss within each group for \(L_g\). The anchor prevents a small, difficult group from wholly dictating updates and protects the main aggregate ranking objective. The anchor is an engineering safeguard for this ranking adaptation, not a result established by the cited paper.

## When to use / avoid
**Use when:**
- Temporal or population-slice evaluation repeatedly identifies the same weak slices under frozen candidate groups.
- The suspected shortcut is concentrated in frequent entities, mature historical periods, dense-history users, or one candidate-group-size regime.
- Every proposed group has enough complete mixed-label candidate groups to estimate a low-variance user-normalized loss.

**Avoid when:**
- The apparent deficit comes from tiny support, changes sign across bootstrap resamples, or is not stable across adjacent time windows.
- Group membership requires future interactions, post-ranking outcomes, future item counts, or any feature unavailable at scoring time.
- Groups are protected attributes or sensitive proxies without an approved fairness, privacy, and governance review.
- The training budget cannot accommodate group-complete batches, per-group loss tracking, and validation-based checkpoint selection.

## Requirements and implementation
1. **Freeze group definitions before training.** Build mutually exclusive assignments from train-available fields only. For example: training-period quartile from event timestamp; history-depth buckets from interactions strictly before the impression; item-frequency buckets from counts accumulated only through that event time; or displayed-candidate-count buckets.
2. **Preserve complete displayed candidate groups.** Never split a user’s displayed candidates across minibatches when computing within-user losses. Assign the group at the displayed-group/context level, rather than allowing each candidate in one ranking set to receive a different DRO group.
3. **Set support gates.** Start with 4–8 groups. Merge or exclude a group from DRO weighting if it has fewer than roughly 500 complete training candidate groups, fewer than 100 mixed-label groups, or too few validation groups for a meaningful uncertainty interval. These thresholds are operational defaults; scale them upward for highly noisy losses.
4. **Use stratified group-complete minibatches.** Sample groups approximately uniformly or with a capped inverse-frequency sampler; then correct the intended objective through the explicit group weights rather than accidental batch prevalence.
5. **Track stable group losses.** For each group, maintain an exponential moving average of user-normalized loss, for example EMA decay 0.9–0.99. Update adversarial weights after each batch or every few batches:
   \[
   q_g \leftarrow \frac{q_g\exp(\eta\,\widetilde L_g)}{\sum_j q_j\exp(\eta\,\widetilde L_j)}.
   \]
   Initialize uniformly, clamp logits or weights to prevent a single noisy group from monopolizing training, and log both instantaneous and EMA loss.
6. **Regularize and select checkpoints conservatively.** Increase L2/weight decay versus the ERM baseline, retain dropout or embedding regularization where applicable, and early-stop using a frozen validation scorecard. The importance of stronger regularization or early stopping for worst-group generalization is directly supported by Sagawa et al. ([mlanthology.org](https://mlanthology.org/iclr/2020/sagawa2020iclr-distributionally/?utm_source=openai))

## Starting configuration and expected effects
Start from a converged anchored ERM ranker and change only the objective:

- \(\alpha\): 0.25 initially; tune 0.10, 0.25, 0.50, 0.75.
- DRO learning rate \(\eta\): 0.01 initially; tune 0.003–0.10 after normalizing group losses to comparable scales.
- EMA decay: 0.95 initially; tune 0.90–0.99.
- Maximum group weight: cap at 3–5 times uniform weight initially, or use a temperature/entropy penalty if weights collapse.
- Weight decay: test 2–10 times the ERM setting while retaining the same optimizer schedule.
- Early stopping: evaluate frequently and choose the earliest checkpoint meeting a prespecified aggregate-metric guardrail while improving the target worst-group metric.

Expected effect: this objective should reduce the training and validation loss concentration in predeclared high-loss groups if those groups capture a real and learnable shift. Aggregate GAUC can remain similar, improve, or decline; nDCG@5 can also move either way because the update reallocates capacity toward hard slices and because the optimized surrogate may not exactly match top-5 ranking. Do not claim success from average GAUC alone. Compare aggregate GAUC and nDCG@5 alongside per-group values, worst-group value, and a support-weighted distribution of group deltas. Sagawa et al. found improved worst-group performance with regularized group DRO in their classification settings, but their reported magnitudes and tasks should not be transferred to within-user ranking. ([mlanthology.org](https://mlanthology.org/iclr/2020/sagawa2020iclr-distributionally/?utm_source=openai))

## Diagnostics and risks
- **Weight collapse:** one group rapidly receives nearly all \(q\). Check support, label mix, duplicate records, impossible-to-rank groups, and loss-scale mismatches before increasing robustness strength.
- **All groups fit in training, weak groups fail in validation:** this is the expected overparameterized-model failure mode emphasized by Sagawa et al.; increase regularization, stop earlier, simplify the model, or lower \(\alpha\). ([mlanthology.org](https://mlanthology.org/iclr/2020/sagawa2020iclr-distributionally/?utm_source=openai))
- **No validation improvement despite changing weights:** the grouping may not describe the causal or predictive shift, the base features may lack the signal needed to improve the group, or the weak-slice finding may be noise.
- **Aggregate regression with modest slice gains:** reduce \(\alpha\), tighten the maximum group weight, or require an aggregate GAUC/nDCG@5 non-inferiority guardrail for checkpoint eligibility.
- **Temporal leakage:** never compute “rare item,” user depth, popularity, or period-relative normalization using the full dataset or future validation/test rows. Materialize these values from causal train prefixes and audit their timestamps.
- **Compute cost:** group-complete stratification can increase padding, reduce effective batch diversity, and require more frequent metric aggregation. Record per-group batch counts and effective sample sizes so optimization artifacts are distinguishable from model effects.

## Cheapest check and clean experiment
**Cheap train-only check:** using the existing ERM model and only training data, partition the final train window into two or three chronological holdout blocks. Freeze the proposed group assignment using causal train-prefix features, then measure per-group user-normalized loss and ranking metrics on each block. Proceed only if the same groups are persistently weak, have sufficient mixed-label support, and do not owe their deficit to a handful of users or candidate sets.

**Clean experiment:** hold architecture, features, data split, group-complete sampler, optimizer schedule, seed set, and checkpoint budget fixed. Compare: (A) the anchored ERM objective, versus (B) the same objective plus group DRO. Predeclare \(\alpha\), \(\eta\), support gates, weight cap, regularization sweep, aggregate guardrail, and worst-group selection metric. Report paired per-candidate-group bootstrap intervals for aggregate GAUC, aggregate nDCG@5, each target-group metric, and the minimum supported-group metric. A follow-up ablation may remove the anchor, but do not combine that change with new features or a new sampler.

## Related cards and sources
**Related cards:** `objective.user_normalized_binary_cross_entropy`, `objective.within_user_ranknet_pairwise_loss`, `objective.bce_lambdarank_ndcg5_hybrid`, `training.group_complete_stratified_minibatching`, `evaluation.stratified_temporal_population_evaluation`, `evaluation.frozen_candidate_group_integrity_audit`, `task.leakage_policy`, `experiment.paired_group_bootstrap_confirmation`, `experiment.anchored_fractional_factorial_ablation`.

**Primary source:** Sagawa, Koh, Hashimoto, and Liang, *Distributionally Robust Neural Networks for Group Shifts: On the Importance of Regularization for Worst-Case Generalization*, ICLR 2020. [O