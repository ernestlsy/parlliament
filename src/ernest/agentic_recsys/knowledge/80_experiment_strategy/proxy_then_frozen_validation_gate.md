# Train-prefix proxy screening before frozen-validation confirmation

## Summary and mechanism
Use a **deterministic, training-only temporal proxy** to eliminate weak ideas before spending access to the official frozen validation metric. Choose a cutoff time \(t_p\) wholly inside the training period: fit candidate configurations on events before \(t_p\), then score the subsequent training-period events using the same canonical displayed candidate groups and metric implementation used by the official evaluation. Promote only a predeclared small set of proxy winners to full-fidelity training and one confirmation on frozen validation.

The purpose is not to replace official validation. It is to reduce adaptive reuse of it: repeatedly proposing changes after observing a fixed holdout can overfit that holdout, a central issue in adaptive data analysis and leaderboard design. ([arxiv.org](https://arxiv.org/abs/1411.2664?utm_source=openai)) A temporal proxy is appropriate only if the earlier-to-later training transition is a useful approximation to the deployment/official-validation transition.

Treat proxy selection as a budget gate: it yields a ranking and failure signal, not a publishable claim of improvement. Expected GAUC and nDCG@5 effects are therefore empirical: a well-aligned proxy should preferentially promote changes that later improve the frozen metrics, but it cannot guarantee either metric improves or that their ordering agrees.

## When to use / avoid
**Use when** the official validation set is fixed and scarce, the interaction log has reliable event timestamps, and many uncertain feature, objective, architecture, or optimization ideas must be screened. It is particularly useful when full training is expensive but a shortened fixed-budget run is materially cheaper.

**Avoid when** the training interval is too short, sparse, or regime-mixed to create enough late-train displayed groups; when labels or candidate groups needed by the proxy are unavailable inside training; or when the official protocol forbids derived validation splits. Do not use a random split as a substitute for a temporal proxy when the target protocol is temporal: it can conceal recency, population, inventory, and exposure shifts.

## Requirements and implementation
1. **Freeze the candidate list first.** Record candidate ID, exact hypothesis, code revision, feature availability rule, seed set, training budget, and promotion rule before running the proxy. Do not add variants in response to proxy or official scores.
2. **Define one temporal boundary.** Start with the latest 15–25% of the training time span as proxy evaluation and the preceding 75–85% as proxy fitting. Adjust the boundary only during one predeclared calibration exercise, then freeze it. Require enough complete displayed groups in both partitions; do not split a displayed group across partitions.
3. **Enforce causality.** For every proxy-evaluated event at time \(t\), construct features, histories, popularity statistics, normalizers, vocabularies, and preprocessing state from data strictly earlier than \(t\). Fit learned transforms only on the proxy-fit prefix.
4. **Preserve group semantics.** Evaluate GAUC and nDCG@5 over the canonical displayed candidate groups, with exactly the same eligibility, tie handling, label definition, and missing-item policy as official evaluation. Score full groups, never sampled or reconstructed alternatives unless the official protocol does so.
5. **Use reduced but fixed fidelity.** For screening, use a fixed cheaper recipe—for example, 25–50% of the normal training steps, a fixed data subsample from the proxy-fit prefix if needed, and 1–2 fixed seeds. Keep architecture-independent controls constant. Full-fidelity confirmation restores the normal training horizon, data prefix, and predeclared seed policy.
6. **Calibrate once.** Run 3–5 known baselines or previously understood changes through both proxy and official validation. Log rank agreement (Spearman or pairwise win agreement), signed deltas versus a common baseline, and disagreement by GAUC versus nDCG@5. This is a local operational diagnostic, not evidence that the proxy is universally valid.
7. **Promote conservatively.** A practical default is the top 1–3 candidates by a predeclared proxy rule, plus an unchanged baseline. If GAUC and nDCG@5 disagree, either require improvement on both, or specify a primary metric and a non-degradation guardrail before screening.
8. **Keep an audit log.** For every run, store split timestamps, event and group counts, group-completeness checks, code/configuration hash, features allowed, compute budget, seed, proxy metrics, official metrics if promoted, and the promotion decision.

## Starting configuration and expected effects
Start with a single cutoff chosen to leave roughly 20% of training time for late-train proxy evaluation, with a minimum group-count threshold chosen from metric stability on the baseline. Use one fixed short-run budget for all candidates, top-2 promotion, and one baseline confirmation. If training variance is visibly large, use two fixed seeds in the proxy and aggregate by mean; do not choose the better seed per candidate.

The likely benefit is **experiment efficiency and reduced official-validation adaptivity**, not an inherently higher GAUC or nDCG@5. Proxy winners may show positive frozen-validation deltas if the proxy reproduces the relevant temporal shift and candidate-group structure. nDCG@5 can disagree with GAUC because it emphasizes ordering near the top of each group; log both rather than assuming a global ranking metric is an adequate top-rank surrogate. Benchmark conclusions can also be sensitive to split and training variation, so retain per-seed and per-group evidence rather than treating a single tiny delta as decisive. ([arxiv.org](https://arxiv.org/pdf/2103.03098?utm_source=openai))

## Diagnostics and risks
- **Leakage signature:** unusually large proxy gains that disappear on frozen validation, especially for recency, popularity, target encoding, or sequence features. Audit every feature timestamp and verify that no late-train aggregate enters proxy-fit examples.
- **Proxy drift signature:** proxy-to-official rank correlation near zero or frequent sign reversals across the calibration baselines. The late-train slice may represent a different population, inventory, exposure policy, or label regime; stop using it as a promotion criterion rather than retuning it after each official result.
- **Metric mismatch signature:** GAUC improves while nDCG@5 declines, or vice versa. Inspect metric contributions by displayed-group size, positive count, activity cohort, and time bucket; retain the predeclared primary/guardrail decision.
- **Compute-confounding signature:** candidates receive unequal steps, data volume, early-stopping behavior, or hyperparameter searches. Make screening fidelity identical; otherwise the gate measures allocation policy rather than the candidate change.
- **Adaptive-proxy risk:** repeatedly moving the cutoff, changing filters, or redefining the promotion rule after results turns the proxy itself into a reused holdout. Freeze it after calibration. The general risk of adaptive holdout reuse is established theoretically; this card's specific split fractions and gate sizes are operational defaults, not sourced optimal values. ([arxiv.org](https://arxiv.org/abs/1411.2664?utm_source=openai))

## Cheapest check and clean experiment
**Cheapest train-only check:** run the unchanged baseline and one deliberately modest, predeclared variant at screening fidelity on the proxy. Verify: (a) all proxy labels follow the cutoff, (b) displayed groups are complete and disjoint across the cutoff, (c) all feature snapshots are causal, (d) metric code matches the official evaluator, and (e) the run manifest is reproducible from hashes. If this check fails, repair the pipeline before screening any idea.

**Clean single-variable experiment:** freeze the model, features, optimizer, seed list, proxy cutoff, candidate groups, and training-step budget. Compare only `baseline objective` versus `baseline objective + one specified loss term` on the proxy. Promote the winner only if it meets the predeclared GAUC/nDCG@5 rule; then train exactly those two configurations at full fidelity and evaluate each once on frozen validation. Report proxy and official deltas side by side, including a disagreement flag; do not alter the loss weight after seeing the official result.

## Related cards and sources
**Related cards:** `task.experiment_protocol`, `task.leakage_policy`, `dataset.inventory_and_splits`, `dataset.population_and_pair_shift`, `evaluation.within_user_metrics`, `evaluation.frozen_candidate_group_integrity_audit`, `evaluation.stratified_temporal_population_evaluation`, `efficiency.successive_halving_experiment_promotion`, `experiment.predeclared_hypothesis_ladder`.

**Primary sources:** Dwork, Feldman, Hardt, Pitassi, Reingold, and Roth, *Preserving Statistical Validity in Adaptive Data Analysis*, arXiv:1411.2664; Dwork et al., *Generalization in Adaptive Data Analysis and Holdout Reuse*, arXiv:1506.02629; Blum and Hardt, *The Ladder: A Reliable Leaderboard for Machine Learning Competitions*, arXiv:1502.04585. These establish the adaptive-evaluation motivation; they do not prescribe this exact recommender-system proxy protocol. ([arxiv.org](https://arxiv.org/abs/1411.2664?utm_source=openai))

### Audited web sources

- Preserving Statistical Validity in Adaptive Data Analysis: <https://arxiv.org/abs/1411.2664?utm_source=openai>
- ACCOUNTING FOR VARIANCE IN MACHINE LEARNING BENCHMARKS: <https://arxiv.org/pdf/2103.03098?utm_source=openai>
