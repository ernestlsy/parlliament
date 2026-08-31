# Diversity-gated rank ensemble decision

## Summary and mechanism

Build a **small, deliberately heterogeneous** final blend from validated models that make meaningfully different mistakes—not from a collection of near-duplicate seeds or checkpoints. The mechanism is simple: each model supplies a within-candidate-group ordering; normalize incompatible score scales by converting scores to groupwise ranks (or, secondarily, groupwise standardized scores), then average with equal weights or a predeclared coarse grid. Rank normalization is appropriate when GAUC and nDCG@5 depend primarily on within-group order rather than probability calibration.

The key assumption is that finalists have independently useful signals. Correlated raw scores alone are not decisive: inspect agreement in *per-group metric contributions*, top-5 overlap, and residual/error patterns. Large historical recommender blends demonstrate that combining distinct predictors can be useful, but they also demonstrate the complexity cost of large blends; this card intentionally constrains the ensemble to a reproducible small set. ([cseweb.ucsd.edu](https://cseweb.ucsd.edu/classes/fa17/cse291-b/reading/ProgressPrize2008_BellKor.pdf?utm_source=openai))

## When to use / avoid

**Use when** several frozen-validation finalists have complementary GAUC versus nDCG@5 behavior, prediction files are in identical canonical order, and serving multiple models is acceptable. Favor distinct inductive biases: for example, dot-product retrieval-style scoring, explicit cross-feature ranking, and causal sequence-history ranking.

**Avoid when** candidates are mostly seed/checkpoint/hyperparameter variants, score files have any missing/nonfinite values or uncertain row alignment, or choosing many weights after repeatedly inspecting frozen validation would amount to adaptive tuning. Do not use this blend to repair leakage, an invalid candidate-group split, or mismatched inference preprocessing.

## Requirements and implementation

1. Freeze a canonical validation table keyed by `(group_id, candidate_id)` and verify an exact one-to-one join for every model artifact. Reject duplicates, omissions, nonfinite predictions, and group-size changes.
2. Compute each model's GAUC, nDCG@5, per-group metric contributions, top-5 sets, and pairwise correlations of groupwise ranks. Also inspect correlation of per-group performance deltas versus a common parent/baseline.
3. Admit only models that are individually credible and not near-identical in diagnostics. A practical starting screen is rank correlation below **0.98** *and* nontrivial disagreement in top-5 membership; treat these as review thresholds, not universal scientific cutoffs.
4. For each group with size \(m\), map a model score to a tie-aware percentile rank, such as \((rank-1)/(m-1)\) when \(m>1\). Use a deterministic tie rule. If ranks are unsuitable because calibrated score magnitudes carry known value, standardize scores within group instead; never mix transforms within one blend.
5. Start with equal weights across two or three admitted models. If weights are allowed, predeclare a coarse simplex grid, e.g. increments of **0.25** for two or three models, select once on the frozen validation set, and record every evaluated combination.
6. Re-evaluate GAUC and nDCG@5 from the blended artifact. Confirm improvement against every included parent using a **paired group bootstrap**: resample groups with replacement, recompute the metric difference on the same sampled groups, and report the interval and the fraction of resamples with positive difference. Bootstrap hypothesis-test methodology for ranked-retrieval metrics is established in information-retrieval evaluation. ([jstage.jst.go.jp](https://www.jstage.jst.go.jp/article/imt/2/4/2_4_1062/_article?utm_source=openai))

## Starting configuration and expected effects

Start with **two models**, equal-weight mean of within-group percentile ranks. Add a third only if it passes the diversity review and improves the locked validation result under the paired group bootstrap. Use 2,000–5,000 bootstrap replicates as an engineering default; increase only if the decision is unstable near zero.

Expected effects are empirical, not guaranteed: a blend may improve GAUC when models disagree on broad pairwise ordering, and may improve nDCG@5 when one model repairs the other's top-of-list errors. It can also trade one metric against the other. Do not claim a fixed uplift magnitude; the outcome depends on candidate-set construction, group-size distribution, labels, and the true dependence among component errors.

## Diagnostics and risks

- **No diversity:** correlations near one, nearly identical top-5 lists, and bootstrap differences centered at zero indicate redundant models. Keep the cheapest parent.
- **Scale pathology:** raw-score averaging changes rankings sharply while rank averaging is stable. Prefer the rank transform unless calibrated magnitudes are explicitly required.
- **Tail harm:** aggregate GAUC rises but nDCG@5 falls, or a small set of large groups dominates the gain. Inspect group-size strata and per-group delta quantiles.
- **Validation overfit:** a highly specific non-equal weight vector wins only marginally. Revert to equal weights or hold out a separate blend-selection split.
- **Leakage/order failure:** implausibly large gains, changed row counts, or a blend that varies under harmless file reordering. Re-run the canonical prediction-artifact and group-integrity audits.
- **Compute/serving cost:** latency, memory, and failure surface increase linearly with component scoring unless shared features or representations are reused. Measure end-to-end inference, not only model forward time.

## Cheapest check and clean experiment

**Cheap train-only check:** on an inner validation fold produced entirely from training-period data, generate canonical prediction artifacts for the two strongest architecturally distinct models. Compare groupwise rank correlation, top-5 overlap, and per-group metric-delta correlation. If all show near-identity, stop before any ensemble search.

**Clean single-variable experiment:** keep data split, candidates, preprocessing, checkpoints, and evaluator fixed. Compare the best parent with exactly one predeclared two-model equal-weight groupwise-rank blend. Use the same groups for paired bootstrap confidence assessment. The only changed variable is the final score-combination rule.

## Related cards and sources

Related IDs:
- `task.prediction_artifact`
- `evaluation.frozen_candidate_group_integrity_audit`
- `evaluation.within_user_metrics`
- `evaluation.probability_calibration_and_ranking_error_audit`
- `experiment.paired_group_bootstrap_confirmation`
- `experiment.predeclared_hypothesis_ladder`
- `architecture.embedding_mlp_ranker`
- `architecture.deep_cross_ranker`
- `architecture.causal_sequence_transformer_ranker`

Primary sources:
- Robert M. Bell, Yehuda Koren, and Chris Volinsky, *The BellKor 2008 Solution to the Netflix Prize* (2008). ([cseweb.ucsd.edu](https://cseweb.ucsd.edu/classes/fa17/cse291-b/reading/ProgressPrize2008_BellKor.pdf?utm_source=openai))
- Tetsuya Sakai, *Evaluating Information Retrieval Metrics Based on Bootstrap Hypothesis Tests* (2007), DOI: `10.11185/imt.2.1062`. ([jstage.jst.go.jp](https://www.jstage.jst.go.jp/article/imt/2/4/2_4_1062/_article?utm_source=openai))

### Audited web sources

- Microsoft Word - BellKorSolution2008.doc: <https://cseweb.ucsd.edu/classes/fa17/cse291-b/reading/ProgressPrize2008_BellKor.pdf?utm_source=openai>
- Evaluating Information Retrieval Metrics Based on Bootstrap Hypothesis Tests: <https://www.jstage.jst.go.jp/article/imt/2/4/2_4_1062/_article?utm_source=openai>
