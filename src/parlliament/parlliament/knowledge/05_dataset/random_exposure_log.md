# Random-exposure log as a separate distribution

## Summary and mechanism
`log_random_4_22_to_5_08_pure.csv` contains 1,186,059 impressions over 2022-04-22 through
2022-05-08. All rows have `is_rand=1`, covering all 27,285 users and all 7,583 videos. It is a
different exposure regime from the standard logs and is not ParLLiaMent's official scoring population.

## When to use / avoid
Use it only for explicitly separated exposure-bias or robustness analysis. Avoid concatenating it
with standard training, fitting preprocessing on it, selecting official hypotheses from its labels,
or substituting it for canonical validation without a deliberate protocol change.

## Requirements and implementation
Keep source/regime indicators and metrics separate. The measured random-log long-view rate is 8.50%
and click rate is 17.62%, versus 31.34% and 44.50% in the contemporaneous standard log. The random
log is overwhelmingly `tab=1` (1,178,025 rows), so regime and tab are strongly entangled.

## Starting configuration and expected effects
The safe default for ParLLiaMent is exclusion from training, train-only screening, and official scoring.
If a separate analysis is authorized, compare score distributions and ranking metrics by exposure
regime without feeding random-period outcomes back into the official model-selection loop.

## Diagnostics and risks
Large prevalence and context shifts make pooled metrics hard to interpret. The filename and
`is_rand` flag establish the supplied regime label but do not by themselves justify a particular
causal estimator. Because its dates overlap validation and test, using its outcomes can contaminate
official evaluation decisions.

## Cheapest check and clean experiment
First produce a read-only distribution report by date, tab, user activity, item, and label. Any later
counterfactual or propensity experiment needs a separately stated estimand, assumptions, and split;
it must not silently alter ParLLiaMent's fixed official objective.

## Related cards and sources
See `dataset.inventory_and_splits`, `dataset.interaction_log_schema`,
`dataset.population_and_pair_shift`, and `task.leakage_policy`. Counts and rates were measured from
the supplied random and standard CSVs.
