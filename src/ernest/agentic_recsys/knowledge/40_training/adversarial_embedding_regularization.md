# Adversarial perturbation regularization for embedding rankers

## Summary and mechanism
Add a worst-case, norm-bounded perturbation to the **dense representations used in the scorer**—at minimum the learned user and video embeddings—and train on both the ordinary long-view ranking loss and the loss under that perturbation. For a batch loss \(L\), compute an embedding-space direction \(g=\nabla_E L\), form a detached perturbation \(r=\epsilon g/(\lVert g\rVert_2+\delta)\), and minimize \(L(E)+\lambda L(E+r)\). This is a first-order approximation to a local minimax objective: the ranker is encouraged to remain accurate when its learned representations move in the loss-increasing direction. Adversarial Personalized Ranking (APR) applied this idea to user and item embedding vectors for personalized ranking. ([arxiv.org](https://arxiv.org/abs/1808.03908?utm_source=openai))

The key assumption is that a small movement in continuous embedding space represents a meaningful local robustness test. Do **not** perturb raw IDs, labels, logged outcomes, or causal aggregate features: IDs are discrete, while perturbing aggregates can obscure their point-in-time semantics and leakage controls. Embedding-level perturbation is also the established adaptation when the original inputs are sparse/discrete. ([arxiv.org](https://arxiv.org/abs/1605.07725?utm_source=openai))

## When to use / avoid
Use after a competitive non-adversarial embedding ranker has converged, especially when seed-to-seed validation variance is high, sparse users/videos degrade disproportionately, or ordinary dropout and weight decay do not control overfit. Avoid it for an underfit baseline, a training path that cannot safely execute the additional gradient/forward computation, or before establishing the normal scale of each embedding table.

## Requirements and implementation
Require learned user and video embeddings, a long-view label, and a ranking objective that can be evaluated twice on the same complete training group. Optionally include dense metadata embeddings, but begin with user and video tables only.

1. Restore a converged baseline checkpoint; retain the optimizer state only if that is your normal fine-tuning protocol.
2. On each minibatch, run the ordinary forward pass and calculate \(L_{base}\).
3. Differentiate \(L_{base}\) with respect to the looked-up user/video embedding activations (or the corresponding selected table rows). Do not apply this inner gradient as an optimizer update.
4. Normalize gradients separately by embedding field, preferably per example/vector: \(r_f=\epsilon_f g_f/(\lVert g_f\rVert_2+10^{-12})\). Detach \(r_f\) so the outer update does not differentiate through adversary construction.
5. Re-score the identical examples/groups with \(e_f+r_f\), calculate \(L_{adv}\), then backpropagate \(L_{base}+\lambda L_{adv}\) once. Preserve all masks, negative samples, candidate sets, group weights, and label eligibility between the two passes.
6. Keep validation and test inference unperturbed.

For shared embedding rows within a batch, activation-level perturbations are usually simpler and avoid ambiguous row-wise aggregation. If perturbing table rows instead, aggregate gradients for repeated IDs before normalizing and ensure only rows present in the batch receive perturbations.

## Starting configuration and expected effects
These are conservative engineering defaults, not universal research-derived optima: begin from the baseline checkpoint with \(\lambda=0.25\), perturb user and video embeddings only, and use per-vector \(L_2\) radii equal to 1%, 3%, and 10% of the median unperturbed embedding norm measured on training batches. Start at 3%; reduce the radius if the adversarial loss immediately dominates or optimization destabilizes. Sweep \(\lambda\in\{0.1,0.25,0.5,1.0\}\) only after identifying a safe radius. Keep the base learning rate initially; if loss spikes persist, lower it before increasing regularization.

GAUC and nDCG@5 may improve, remain flat, or fall: do not assume a gain or transfer published magnitudes to this data, objective, or candidate distribution. A desirable result is a reproducible validation improvement or comparable primary metrics with lower seed variance and a smaller adversarial-loss gap. APR provides evidence that adversarial training can improve ranking robustness and generalization for matrix-factorization BPR, but it does not establish a guaranteed effect for deep long-view rankers or listwise losses. ([arxiv.org](https://arxiv.org/abs/1808.03908?utm_source=openai))

## Diagnostics and risks
**Leakage:** compute embedding-norm statistics, checkpoint selection, and hyperparameter choices from training data only. Maintain as-of timestamps for metadata and causal behavior/popularity features; adversarial noise must not cause a fallback path that reads post-impression or post-label aggregates.

**Compute:** the method needs an additional gradient construction plus a perturbed scoring pass. Profile peak memory, step time, and failed/overflowed steps before a full run; reduce batch size only if this does not change group completeness or negative-sampling behavior.

**Failure signatures:** (a) \(L_{adv}\) is nearly identical to \(L_{base}\): radius is too small, gradients are detached too early, or perturbations are not reaching the scorer; (b) \(L_{adv}\) is enormous from step one: radius scaling is wrong or fields need separate radii; (c) training GAUC falls while validation metrics do not recover: regularization is too strong; (d) GAUC rises while nDCG@5 falls: inspect top-rank calibration, candidate/group composition, and the objective trade-off; (e) sparse-entity outcomes worsen: compare perturbation-to-embedding-norm ratios by frequency bucket rather than using one unscaled absolute radius.

Monitor base loss, adversarial loss, their difference, per-field gradient norms, per-field perturbation/embedding norm ratios, GAUC, nDCG@5, and results by user/video frequency bucket. Also check exposure/popularity concentration: adversarial ranking can have beyond-accuracy trade-offs, so accuracy-only acceptance is insufficient.

## Cheapest check and clean experiment
**Cheap train-only check:** freeze model parameters on several fixed training batches. Construct perturbations once and verify that \(L_{adv}>L_{base}\) for most batches, that perturbation norms equal their configured bounds, and that replacing adversarial directions with same-norm random directions produces a smaller average loss increase. This validates sign, normalization, attachment point, and masking without selecting on validation data.

**Clean experiment:** run the established baseline and one adversarial variant with identical data split, feature snapshots, candidate construction, minibatch/group schedule, optimizer schedule, stopping rule, and at least three fixed seeds. Change only the adversarial term. Pre-register the radius and \(\lambda\) using a small training/validation pilot, then report mean and dispersion for GAUC and nDCG@5, frequency slices, wall-clock cost, and the adversarial-loss gap. If tuning is necessary, use a bounded grid and compare the final selected configuration with a baseline that received an equivalent tuning budget.

## Related cards and sources
Related cards: `features.entity_id_embeddings`, `features.causal_behavior_history_features`, `features.train_prefix_smoothed_popularity_features`, `architecture.embedding_mlp_ranker`, `objective.within_user_ranknet_pairwise_loss`, `objective.bce_lambdarank_ndcg5_hybrid`, `training.group_complete_stratified_minibatching`, `task.leakage_policy`, `evaluation.within_user_metrics`, `task.experiment_protocol`.

Primary sources: He, He, Du, and Chua, *Adversarial Personalized Ranking for Recommendation*, SIGIR 2018, DOI: `10.1145/3209978.3209981` (also arXiv: `1808.03908`). ([arxiv.org](https://arxiv.org/abs/1808.03908?utm_source=openai)) Miyato, Dai, and Goodfellow, *Adversarial Training Methods for Semi-Supervised Text Classification*, arXiv: `1605.07725`; it motivates perturbing continuous embeddings rather than sparse discrete inputs. ([arxiv.org](https://arxiv.org/abs/1605.07725?utm_source=openai))

### Audited web sources

- Adversarial Personalized Ranking for Recommendation: <https://arxiv.org/abs/1808.03908?utm_source=openai>
- Adversarial Training Methods for Semi-Supervised Text Classification: <https://arxiv.org/abs/1605.07725?utm_source=openai>
