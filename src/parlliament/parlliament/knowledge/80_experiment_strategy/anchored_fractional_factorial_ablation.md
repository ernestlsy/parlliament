# Anchored fractional-factorial ablation plan

## Summary and mechanism
Anchor every run to one versioned reference ranker, then represent candidate interventions as binary factors—for example: causal history, train-prefix aggregate features, ranking loss, and auxiliary supervision. Run a compact, approximately orthogonal screening matrix rather than a sequence of cumulative additions. Estimate main effects first from contrasts against the same anchor; spend follow-up runs only on interactions that are mechanistically plausible, such as history representation × ranking loss.

This design assumes each factor can be enabled or disabled without changing the meaning of the others, and that all comparisons share the same data snapshot, feature-availability policy, candidate groups, budget, checkpoint rule, and seed block. It improves attribution, not necessarily statistical power: use repeated seeds or paired group-level uncertainty estimates before declaring a small effect real. Keep the evaluation timeline causal. Offline recommender results can be distorted when training uses interactions or item availability from after a test event. ([arxiv.org](https://arxiv.org/abs/2010.11060?utm_source=openai))

## When to use / avoid
Use when several independently toggleable changes compete for limited training budget and naïve stacking would confound their contributions. Avoid when a component is invalid alone, when the baseline is nondeterministic, or when fewer than a baseline plus a minimal screening block can be afforded.

## Requirements and implementation
1. Freeze an anchor: code commit, data/split hashes, candidate-group construction, optimizer schedule, max steps, early-stop/checkpoint selection rule, and inference settings.
2. Publish a feature manifest for every factor: source table, event-time cutoff, fit population, missing-value rule, and whether it is available at serving time.
3. Start with four binary factors. Use an 8-run half-fraction for screening plus the anchored all-off run; choose the generator before observing results and record factor aliases. If compute permits, use the full 16-run design or add foldover runs to resolve aliased effects.
4. Use one fixed seed for the screening pass only if necessary; promote promising contrasts to 3–5 fixed, predeclared seeds. Preserve seed identity across compared configurations.
5. Store a ledger row per run: parent anchor, factor vector, hashes, wall-clock/GPU cost, training status, GAUC, nDCG@5, per-group metrics, and artifact paths.
6. Reserve 2–4 runs for interactions. Default candidates: causal history × ranking loss; aggregate features × auxiliary supervision. Do not search all pairwise interactions after screening.

## Starting configuration and expected effects
Use factors `H` = causal behavior history, `A` = train-prefix smoothed aggregate features, `L` = within-user ranking-aware loss, and `M` = shared-trunk auxiliary behavior supervision. Begin with an embedding-MLP anchor trained with user-normalized BCE. Evaluate GAUC and nDCG@5 on frozen candidate groups.

Expected metric directions are empirical, not guaranteed: `L` may be more visible in nDCG@5 than GAUC because it directly changes within-group ordering; `H` may help either metric when recent behavior is predictive; `A` may help cold or sparse entities but can fail under temporal shift; `M` can regularize shared representations or cause negative transfer. Do not assign expected effect sizes in advance. Exposure-conditioned logs also mean offline improvements need not estimate preference or online lift without suitable exposure information or a defensible counterfactual design. ([proceedings.neurips.cc](https://proceedings.neurips.cc/paper/2020/hash/9cd013fe250ebffc853b386569ab18c0-Abstract.html?utm_source=openai))

## Diagnostics and risks
- **Leakage signature:** unusually large lift from aggregates or histories, especially concentrated near split boundaries. Audit every feature timestamp and recompute using only each example’s prefix.
- **Alias/confounding signature:** two factors appear beneficial in the fraction, but a foldover reverses or separates them. Report the estimand as an aliased contrast until resolved.
- **Budget confounding:** one factor changes parameter count, convergence speed, or usable batch size. Match steps and data exposure; additionally report compute-normalized results.
- **Checkpoint fishing:** a treatment wins only because it had more evaluated checkpoints. Apply one identical, validation-only checkpoint rule to every arm.
- **Interaction failure:** a factor is neutral alone but harmful in combination. Inspect training loss, calibration, group-size strata, and user/activity slices before expanding the design.

## Cheapest check and clean experiment
**Cheap train-only check:** construct the toggle matrix; for every row, assert identical split hashes, seed, step budget, candidate-group IDs, and feature timestamps. Then run the anchor and one `H`-only arm for a short fixed prefix of training. Compare loss curves, feature missingness, score distributions, and exact reproducibility after rerun; do not use this check for model selection.

**Clean single-variable experiment:** run anchor (`H=A=L=M=0`) versus history-only (`H=1, A=L=M=0`) with identical seed block, training budget, checkpoint rule, and frozen evaluation candidates. Report per-seed GAUC and nDCG@5 deltas, aggregate uncertainty, training cost, and temporal-slice results. Only after this contrast is stable should `H` enter the fractional screen or an `H × L` interaction test.

## Related cards and sources
Related: `task.experiment_protocol`, `task.leakage_policy`, `features.causal_behavior_history_features`, `features.train_prefix_smoothed_popularity_features`, `objective.within_user_ranknet_pairwise_loss`, `objective.shared_trunk_auxiliary_behavior_multitask_loss`, `evaluation.frozen_candidate_group_integrity_audit`, `evaluation.stratified_temporal_population_evaluation`, `experiment.predeclared_hypothesis_ladder`, `experiment.paired_group_bootstrap_confirmation`.

Primary sources: Ji, Sun, Zhang, and Li, *A Critical Study on Data Leakage in Recommender System Offline Evaluation*, arXiv:2010.11060. Xu et al., *Adversarial Counterfactual Learning and Evaluation for Recommender System*, NeurIPS 2020. Swaminathan et al., *Off-policy Evaluation for Slate Recommendation*, NeurIPS 2017.

### Audited web sources

- A Critical Study on Data Leakage in Recommender System Offline Evaluation: <https://arxiv.org/abs/2010.11060?utm_source=openai>
- Adversarial Counterfactual Learning and Evaluation for Recommender System: <https://proceedings.neurips.cc/paper/2020/hash/9cd013fe250ebffc853b386569ab18c0-Abstract.html?utm_source=openai>
