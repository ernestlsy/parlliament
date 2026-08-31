# Validation-safe metric-aligned checkpoint averaging

## Summary and mechanism
Save model checkpoints on a **fixed, predeclared late-training schedule**, score every saved checkpoint once using the primary validation composite, choose one small *contiguous* window by a deterministic rule, and emit predictions from the arithmetic mean of the compatible model parameters in that window. For checkpoints \(\theta_1,\ldots,\theta_K\), use \(\bar\theta=K^{-1}\sum_i\theta_i\). The intended effect is to reduce sensitivity to stochastic movement among nearby late-training solutions while retaining one inference-time model.

This is a constrained, validation-governed variant of stochastic weight averaging rather than an ensemble: it averages weights from one training trajectory, so serving cost is normally unchanged. Weight averaging has established roots in iterate averaging for stochastic approximation, and SWA showed that averaging SGD trajectory points can improve generalization and reach wider solutions under its studied schedules. ([epubs.siam.org](https://epubs.siam.org/doi/10.1137/0330046?utm_source=openai)) This card does **not** imply a guaranteed improvement for ranking metrics or for every optimizer trajectory.

## When to use / avoid
**Use when:** late validation GAUC and nDCG@5 oscillate modestly; checkpoints share exactly the same architecture, parameter names, tensor shapes, feature schema, and preprocessing; and a single run must be made more robust without a broad sweep.

**Avoid when:** validation curves show sustained late decline, numerical instability, or a regime break from a learning-rate restart; checkpoints cross architecture or embedding-vocabulary changes; or the window is repeatedly redesigned after inspecting secondary metrics, test results, or many validation slices. Do not average independently trained seeds unless that operation was separately validated; parameter correspondence is not assured merely because shapes match.

## Requirements and implementation
1. Freeze the validation split, user/group definition, prediction row order, candidate order, and primary composite before training. Keep an immutable checkpoint manifest containing epoch, global step, training-data cutoff, code/config hash, and validation score.
2. Predeclare a late checkpoint schedule, for example every epoch over the final 20% of training, or every fixed number of optimizer steps after the last planned learning-rate decay. Save only model state needed for inference; retain optimizer state separately if resumption is needed.
3. Predeclare one primary composite, e.g. `0.5 * standardized_validation_GAUC + 0.5 * standardized_validation_nDCG@5`, or use the competition/project composite exactly as specified. Standardization constants must be fixed from prior experiments or omitted; never estimate them using the current test set.
4. Predeclare selection: among contiguous windows of width `K`, select the window with the greatest mean primary validation composite. Break exact ties by the earliest window. Evaluate each checkpoint once; do not choose a different window for each reported metric.
5. Average floating-point parameters tensorwise in FP32, then cast only for the final artifact if required. Exclude counters and non-parameter metadata unless their semantics are explicitly compatible. For batch normalization, recompute running statistics by a forward-only pass over permitted **training** data after averaging, or do not use direct averaging if this cannot be done safely.
6. Generate one canonical validation/test prediction artifact from the averaged model, preserving the required fixed row order. Record its source checkpoint IDs and coefficients.

Data leakage boundary: validation labels may select the one predeclared window, but must not update weights, normalization statistics, feature aggregates, calibration, thresholds, or checkpoint schedules. Test labels must never participate. Training-data cutoffs and causal feature construction must remain identical across all averaged checkpoints.

## Starting configuration and expected effects
Start with `K=3` adjacent late checkpoints, saved one epoch apart (or at a cadence that captures visible late-trajectory variation). If the trajectory is smooth and stable, try `K=5`; rarely exceed 7 without evidence that the full span stays in one optimization basin. Fix equal weights first. A useful fixed candidate policy is: save the final 10 checkpoints; consider only widths 3 and 5; choose the best mean-composite window using the stated earliest-window tie rule. If comparing widths, that comparison itself consumes validation-selection budget, so declare it once for the whole experiment family rather than tailoring it per run.

Likely effects are empirical: GAUC may become less variable when pairwise ordering is sensitive to late parameter noise, while nDCG@5 may improve, remain unchanged, or worsen because top-rank ordering is more locally discontinuous. Do not claim a magnitude without repeated held-out evidence. The most credible outcome is often reduced run-to-run fragility rather than a large mean lift. SWA research supports the general possibility of improved generalization from trajectory weight averaging, but it does not establish a specific GAUC or nDCG@5 gain for this ranking setup. ([mlanthology.org](https://mlanthology.org/uai/2018/izmailov2018uai-averaging/?utm_source=openai))

## Diagnostics and risks
**Healthy signature:** selected checkpoints are neighboring in time, their validation composites are similar, averaged-model validation performance is near or above the constituent-window mean, and score distributions remain well formed.

**Failure signatures:** (a) the averaged model is sharply worse than every constituent checkpoint, suggesting incompatible modes, a restart boundary, or state-handling error; (b) GAUC rises while nDCG@5 falls materially, suggesting changed head ordering; (c) results depend on prediction row order, indicating an evaluation/artifact bug; (d) batch-normalization models degrade only after averaging, suggesting stale running statistics; (e) the selected window repeatedly moves after auxiliary-slice inspection, indicating validation overfitting.

Compute cost is low for arithmetic averaging but not zero: checkpoint storage, one validation pass per scheduled checkpoint, optional normalization-statistic recomputation, and a final prediction pass are required. Limit the checkpoint count in advance rather than evaluating dense checkpoints until a favorable window appears.

## Cheapest check and clean experiment
**Cheapest train-only check:** after training, average three adjacent late checkpoints and run a forward pass on a fixed, label-free training-feature batch. Verify identical parameter keys/shapes, finite averaged tensors, deterministic output order, finite logits, and no material collapse in score variance relative to the three source models. This catches serialization, compatibility, and numerical failures without spending additional validation selection budget.

**Clean single-variable experiment:** hold seed, data split, checkpoint schedule, optimizer schedule, features, architecture, training steps, metric code, and prediction writer fixed. Compare: (A) the predeclared single-checkpoint rule versus (B) the predeclared `K=3` contiguous averaging rule. Select each only through the same primary validation composite and tie rule, then produce exactly one untouched test artifact per condition. Repeat over predeclared seeds if experiment budget permits; report per-seed GAUC and nDCG@5 plus the mean and dispersion, rather than selecting the better method separately for each seed.

## Related cards and sources
**Related cards:** `task.experiment_protocol`, `task.leakage_policy`, `task.prediction_artifact`, `evaluation.within_user_metrics`, `training.group_complete_stratified_minibatching`, `architecture.embedding_mlp_ranker`, `architecture.deep_cross_ranker`.

**Primary sources:** Polyak & Juditsky, *Acceleration of Stochastic Approximation by Averaging*, SIAM Journal on Control and Optimization (1992), DOI: `10.1137/0330046`. Izmailov et al., *Averaging Weights Leads to Wider Optima and Better Generalization*, UAI (2018), arXiv: `1803.05407`. ([epubs.siam.org](https://epubs.siam.org/doi/10.1137/0330046?utm_source=openai))

### Audited web sources

- Acceleration of Stochastic Approximation by Averaging | SIAM Journal on Control and Optimization: <https://epubs.siam.org/doi/10.1137/0330046?utm_source=openai>
- Averaging Weights Leads to Wider Optima and Better Generalization | ML Anthology: <https://mlanthology.org/uai/2018/izmailov2018uai-averaging/?utm_source=openai>
