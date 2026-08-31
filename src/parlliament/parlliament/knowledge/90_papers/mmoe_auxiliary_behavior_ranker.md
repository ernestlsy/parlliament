# Multi-gate mixture-of-experts auxiliary-behavior ranker

## Summary and mechanism

Replace a fully shared multitask trunk with **multi-gate mixture-of-experts (MMoE)**: several compact shared experts transform the same impression-time feature vector, while the long-view task and every admitted auxiliary behavior each learn their own softmax gate over those experts and their own prediction tower. For task \(t\), the representation is \(h_t(x)=\sum_e g_{t,e}(x) f_e(x)\). The long-view head remains the deployment objective; auxiliary heads supply additional gradients but do not need to share one identical representation.

The key assumption is that tasks have some useful common structure but differ enough that forcing all of them through one shared bottom can cause negative transfer. The original MMoE paper introduces task-specific gates over shared experts, reports that the architecture explicitly models task relationships, and finds advantages over compared multitask baselines when tasks are less related; it also demonstrates the approach in a large-scale content-recommendation setting. ([doi.org](https://doi.org/10.1145/3219819.3220007?utm_source=openai))

Use only labels whose observation window and features are valid at impression time. An auxiliary label may occur after the impression; that is acceptable as a target, but no feature may encode future behavior, later exposure, or outcome-derived state unavailable when the item was ranked.

## When to use / avoid

**Use when:**

- At least one auxiliary binary behavior has adequate prevalence, stable definition, and an impression-time-valid label window.
- A shared-trunk auxiliary-behavior model gives neutral, unstable, or segment-dependent long-view results.
- Long-view and auxiliary behaviors plausibly require different combinations of user, item, context, and causal-history signals.
- A small multi-head change is preferable to separately training and serving several rankers.

**Avoid when:**

- Long-view is the only trustworthy label, or auxiliary positives are so sparse that their head is dominated by sampling noise.
- The auxiliary target or any of its features violates the leakage/availability policy.
- The serving or training budget cannot accommodate several expert forward passes and task heads.
- There is no clean way to preserve the long-view objective as dominant; optimizing an easy proxy can improve its own AUC while harming top-ranked long-view quality.

## Requirements and implementation

1. Start from the exact long-view feature pipeline, including only features materialized as of the ranking timestamp. Add one label column per admitted auxiliary behavior, with explicit eligibility, attribution, and censoring rules.
2. Build a compact expert bank over the existing dense/sparse feature representation. Each expert can be a two-layer MLP with the same output width. Gates take the same valid input representation and produce a softmax distribution over experts for each task.
3. Give every task a separate gate and lightweight tower. Do not share a final logit layer: task-specific calibration and decision boundaries are part of the point of the architecture.
4. Train on examples carrying all labels where possible. If labels have different eligibility populations, mask each task loss outside its eligible rows rather than creating implicit negatives.
5. Use a long-view-dominant objective, for example \(L=L_{long}+\sum_a \lambda_a L_a\), where \(L_{long}\) is the current user-normalized BCE or ranking loss and \(L_a\) is masked BCE for auxiliary task \(a\). Normalize losses by valid-label count and, when applicable, retain within-user/group normalization for the primary task.
6. Log per-task loss, head calibration, gate entropy, mean gate weights, gate distributions by cohort, expert activation norms, and gradient norms into shared experts. These are diagnostics, not optimization targets.

Data checks should precede modeling: verify label timestamps, feature availability joins, duplicate impressions, target-window truncation near split boundaries, and whether a missing auxiliary label means ineligible, unknown, or negative. Freeze candidate groups and temporal splits before comparing against the baseline.

## Starting configuration and expected effects

**Practical starting point (empirical defaults, not claims from the paper):**

- Experts: 2–4.
- Expert MLP: 1–2 hidden layers; width equal to, or one-half of, the baseline trunk width.
- Gate: linear projection from the shared input representation to expert logits; softmax over experts.
- Towers: one small hidden layer or linear head initially.
- Auxiliary tasks: begin with one or two, not every available engagement event.
- Auxiliary weights: begin at 0.05–0.25 each after accounting for loss scale; tune over 0.01, 0.05, 0.10, 0.25, and 0.50 while keeping the primary coefficient fixed at 1.0.
- Regularization: reuse baseline embedding and MLP regularization; add modest gate-logit regularization only if gates saturate prematurely. Do not force uniform expert use by default.

Expected outcome: GAUC and nDCG@5 can improve if auxiliaries add signal and the gates separate incompatible sharing patterns. No magnitude should be assumed: the cited paper establishes an architectural rationale and empirical improvements in its settings, not a transferable uplift guarantee for this dataset or metric. ([doi.org](https://doi.org/10.1145/3219819.3220007?utm_source=openai)) A common outcome is improved auxiliary prediction with unchanged or degraded long-view nDCG@5; treat that as evidence that task weighting, label quality, or task relatedness needs review rather than as success.

## Diagnostics and risks

- **Negative transfer persists:** long-view GAUC/nDCG@5 falls while auxiliary metrics rise. Reduce or remove the offending auxiliary, lower its weight, inspect its eligibility population, and compare with a primary-only MMoE to separate architecture cost from auxiliary-gradient cost.
- **Gate collapse:** nearly all rows and tasks select the same expert, or gate entropy quickly approaches zero. Check initialization, gate learning rate, and whether experts are too small or too similar. A collapsed model may simply be an expensive shared trunk.
- **Unhelpful separation:** gates differ strongly by task but primary ranking does not improve. This can indicate insufficient shared signal, noisy auxiliary labels, or an over-parameterized model.
- **Expert underuse or instability:** one expert has persistently small activation/gradient norms, or expert usage changes sharply across seeds. Reduce expert count, increase data per parameter, or simplify towers.
- **Leakage:** large offline lifts concentrated in rows with delayed metadata, post-impression histories, or split-boundary labels are a warning. Audit every join against the impression timestamp and rebuild causal histories by train/validation/test prefix.
- **Compute regression:** dense MMoE evaluates every expert for every candidate. Measure candidate-level latency, memory, throughput, and feature-fetch cost against the shared-trunk baseline; compact experts are preferable to adding capacity indiscriminately.
- **Metric mismatch:** better pointwise primary BCE but worse nDCG@5 suggests recalibrating the primary ranking component, candidate-group sampling, or checkpoint selection—not increasing auxiliary weight automatically.

## Cheapest check and clean experiment

**Cheap train-only check:** on the frozen training period, fit the shared-trunk baseline and a same-parameter-budget MMoE with the *primary loss only*. Compare training/held-out-train-prefix primary loss, gate entropy, expert utilization, and seed stability. If primary-only MMoE is less stable or offers no representation benefit, do not attribute later changes to auxiliary labels. Next, compute label prevalence, co-occurrence, and conditional prevalence by user/activity cohorts for each candidate auxiliary; reject labels with unclear eligibility or timestamp provenance.

**Clean single-variable experiment:** preserve features, candidates, temporal split, minibatches, optimizer schedule, primary objective, total parameter budget as closely as possible, and checkpoint rule. Compare: (A) the existing shared-trunk multitask model versus (B) MMoE with the same task heads and loss weights. Run multiple fixed seeds and report paired, within-user GAUC and nDCG@5 differences with uncertainty. Then test one predeclared auxiliary-weight change at a time. Select only on frozen validation; evaluate the chosen configuration once on the untouched test period.

## Related cards and sources

Relevant cards: `task.leakage_policy`, `dataset.interaction_log_schema`, `dataset.inventory_and_splits`, `features.causal_behavior_history_features`, `architecture.embedding_mlp_ranker`, `objective.user_normalized_binary_cross_entropy`, `objective.within_user_ranknet_pairwise_loss`, `objective.shared_trunk_auxiliary_behavior_multitask_loss`, `training.group_complete_stratified_minibatching`, `evaluation.within_user_metrics`, `evaluation.frozen_candidate_group_integrity_audit`, `evaluation.stratified_temporal_population_evaluation`, `evaluation.probability_calibration_and_ranking_error_audit`, `experiment.paired_group_bootstrap_confirmation`, `experiment.proxy_then_frozen_validation_gate`.

Primary source: Ma, J., Zhao, Z., Yi, X., Chen, J., Hong, L., and Chi, E. H. (2018), *Modeling Task Relationships in Multi-task Learning with Multi-gate Mixture-of-Experts*, KDD 2018, DOI: `10.1145/3219819.3220007`. ([doi.org](https://doi.org/10.1145/3219819.3220007?utm_source=openai))

### Audited web sources

- Modeling Task Relationships in Multi-task Learning with Multi-gate Mixture-of-Experts | Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining: <https://doi.org/10.1145/3219819.3220007?utm_source=openai>
