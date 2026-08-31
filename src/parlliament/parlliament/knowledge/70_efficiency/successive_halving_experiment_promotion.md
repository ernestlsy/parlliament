# Predeclared successive-halving experiment promotion

## Summary and mechanism

Successive halving allocates a small, fixed training budget to every **predeclared** candidate configuration, ranks candidates using one frozen validation protocol, eliminates a fixed fraction, and spends larger budgets only on the survivors. For an official experiment, make this a deterministic, auditable procedure rather than an informal early-stopping practice: fix the candidate table, rung budgets, seeds, validation groups, metric definitions, composite ranking rule, tie-breaks, and promotion counts before any runs begin.

The premise is that low-fidelity performance—such as early-epoch validation GAUC and nDCG@5—has enough agreement with full-fidelity performance to remove clearly weak candidates. Successive halving was formulated for iterative learning procedures as a non-stochastic best-arm-identification problem; it can improve resource use when promising configurations can be distinguished before convergence, but its value depends on the behavior of the learning curves. ([proceedings.mlr.press](https://proceedings.mlr.press/v51/jamieson16.html?utm_source=openai)) Hyperband generalized this adaptive-resource-allocation idea across multiple resource brackets; this card deliberately uses one predeclared bracket for easier experimental accounting and reproducibility. ([jmlr.org](https://www.jmlr.org/beta/papers/v18/16-558.html?utm_source=openai))

For rung \(r\), train each surviving configuration \(c\) to cumulative budget \(b_r\), score it on the same frozen validation groups, sort by a fixed composite score, and retain the top \(k_{r+1}\). Never replace an eliminated candidate, alter validation composition, or revise weights after observing outcomes.

## When to use / avoid

**Use when:** several credible architecture, objective, or regularization variants compete; official experiments are limited; and prior pilot evidence suggests that low-budget and full-budget rankings are sufficiently aligned. This is especially useful where each candidate can be resumed faithfully from a rung checkpoint.

**Avoid when:** early learning curves cross frequently; sequence models need long warm-up before their history encoder becomes useful; auxiliary-task benefits arrive late; or a low-fidelity rung would change leakage handling, candidate-group construction, feature availability, or the definition of GAUC/nDCG@5. Do not use it where the experiment protocol requires equal training allocation for every candidate.

## Requirements and implementation

1. **Freeze the candidate table.** Assign each configuration a stable ID and record all architecture, feature, objective, optimizer, schedule, batch, regularization, initialization, and data-version settings. Do not create follow-up variants after seeing rung results.
2. **Freeze data and evaluation.** Use identical temporal cutoffs, eligibility rules, candidate groups, leakage controls, and validation examples at every rung. A smaller validation subset is permissible only if it is frozen before execution and preserves complete evaluation groups.
3. **Use deterministic seeds.** Fix a training seed, data-order seed, negative-sampling seed, and any distributed-runtime determinism settings for each candidate. Prefer the same seed tuple across candidates within a screening bracket; log nondeterministic-kernel exceptions explicitly.
4. **Define cumulative budgets.** A practical starting schedule is three rungs with \(b=[0.25,0.5,1.0]\) of the normal training budget and retention \(k=[N,\lceil N/3\rceil,\lceil N/9\rceil]\). For highly unstable early curves, start later, for example \(b=[0.4,0.7,1.0]\), and retain more candidates. Budgets can be epochs, optimizer steps, examples processed, or wall-clock time, but must be comparable across configurations.
5. **Define one fixed composite promotion rule.** To avoid one metric's scale dominating, rank candidates separately by validation GAUC and nDCG@5, then use \(S=0.5\,\mathrm{rank}_{GAUC}+0.5\,\mathrm{rank}_{nDCG@5}\), with lower \(S\) better. Predeclare different weights only if the product objective justifies them. Break exact ties by higher nDCG@5, then higher GAUC, then lexicographic configuration ID.
6. **Resume, do not restart, between rungs** unless the protocol explicitly defines independent reruns. Preserve optimizer state, scheduler state, RNG state where possible, and checkpoint identity. At the final rung, rerun or confirm the promoted configuration(s) at full training fidelity and score them using the full frozen validation protocol.
7. **Log an append-only decision ledger.** For every candidate and rung record: configuration hash; code/data/container hashes; seed tuple; start and end timestamps; allocated and consumed budget; checkpoint URI/hash; GAUC; nDCG@5; composite score; rank; promotion decision; and the predeclared rule version.

## Starting configuration and expected effects

Start with \(N=9\) to \(27\) candidates, a three-rung 3× retention ratio, and one deterministic seed per candidate when the purpose is screening rather than estimating seed variance. Reserve a separate confirmation run with 2–3 fixed seeds for the final winner only if the counted-experiment policy permits it. If only one final configuration may be fully trained, promote two candidates to the last rung when feasible; a close low-fidelity race is a warning that screening uncertainty is material.

The expected effect is primarily **greater candidate coverage under a fixed compute or experiment budget**, not a guaranteed improvement in either metric. If the low-fidelity ranking agrees with the full-fidelity ranking, promotion can reach a competitive configuration while avoiding full training of weak alternatives. Established work reports favorable compute savings for adaptive allocation in some hyperparameter-optimization settings, but it does not establish a universal GAUC or nDCG@5 gain for recommender ranking tasks. ([proceedings.mlr.press](https://proceedings.mlr.press/v51/jamieson16.html?utm_source=openai)) Therefore, treat any GAUC or nDCG@5 improvement as an empirical result of the specific dataset, model family, and rung design—not as an expected magnitude.

## Diagnostics and risks

**Key diagnostic:** on a completed pilot set, compute Spearman rank correlation between rung-0 composite ranks and full-fidelity composite ranks, plus top-\(k\) recall: whether the final top candidate was retained at each rung. Predeclare a minimum acceptable retention criterion from historical pilots; do not retrofit it from the official run.

**Failure signatures:**

- Large rank reversals or poor top-\(k\) recall: the first budget is too short or the candidate family has delayed gains. Increase the first-rung budget, reduce elimination aggressiveness, or abandon halving.
- High variance among near-tied candidates: promote a wider set, use deterministic execution, or require a confirmation seed set at the final rung.
- Early GAUC rises while nDCG@5 later diverges: the composite or rung budget is not aligned with the product metric; inspect per-metric ranks rather than only the aggregate.
- A promoted model cannot reproduce its rung score after resume: checkpoint, RNG, data-order, or preprocessing state was not preserved; invalidate that bracket rather than silently continuing.
- Apparent gains only after changing validation rows, candidate groups, or feature snapshots: this is a protocol change or leakage risk, not evidence for promotion.

Compute accounting must include failed jobs, checkpoint storage, validation passes, and reruns. The adaptive allocation itself is legitimate only if the allocation rule was fixed in advance and every elimination is retained in the ledger.

## Cheapest check and clean experiment

**Cheap train-only check:** before touching frozen validation, execute all candidate configurations for a tiny fixed number of steps on the identical training-data prefix. Verify configuration hashing, seed reproducibility, equal budget enforcement, checkpoint/resume equivalence, metric logging schema, deterministic tie-breaking, and that the scheduler promotes exactly the precomputed IDs when supplied synthetic scores. This validates mechanics; training loss is not evidence that low-fidelity validation ranks will predict final ranking quality.

**Clean single-variable experiment:** hold the candidate set, total nominal training budget, frozen validation groups, seeds, optimizer schedules, final full-fidelity budget, and composite metric fixed. Compare: (A) uniform allocation, training every candidate to full budget; versus (B) the predeclared successive-halving schedule. Evaluate the selected configuration once on an untouched final evaluation split after selection is complete. Report compute consumed, number of fully trained candidates, rung-wise rank correlation, retention of the uniform winner, final GAUC, final nDCG@5, and every deviation or failed run. The sole manipulated variable is allocation policy.

## Related cards and sources

**Related cards:** `task.experiment_protocol`, `task.leakage_policy`, `dataset.inventory_and_splits`, `evaluation.frozen_candidate_group_integrity_audit`, `evaluation.within_user_metrics`, `evaluation.stratified_temporal_population_evaluation`, `training.metric_aligned_checkpoint_averaging`.

**Primary sources:** Kevin Jamieson and Ameet Talwalkar, *Non-stochastic Best Arm Identification and Hyperparameter Optimization*, AISTATS 2016, PMLR 51:240–248. ([proceedings.mlr.press](https://proceedings.mlr.press/v51/jamieson16.html?utm_source=openai)) Lisha Li, Kevin Jamieson, Giulia DeSalvo, Afshin Rostamizadeh, and Ameet Talwalkar, *Hyperband: A Novel Bandit-Based Approach to Hyperparameter Optimization*, JMLR 18(185):1–52, 2018. ([jmlr.org](https://www.jmlr.org/beta/papers/v18/16-558.html?utm_source=openai))

### Audited web sources

- Non-stochastic Best Arm Identification and Hyperparameter Optimization: <https://proceedings.mlr