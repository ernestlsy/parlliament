# Within-user ranking and diagnostic metrics

## Summary and mechanism
GAUC summarizes per-user positive-negative ordering with positive-count weighting; nDCG@5 emphasizes
discounted relevance at the top five. Primary averages them.

## When to use / avoid
Use official metrics for selection and stopping. Use accuracy, precision, recall, F1, global AUC,
average precision, MAP@5, MRR@5, and HitRate@5 only to diagnose mechanisms.

## Requirements and implementation
Preserve canonical rows and group only inside fixed evaluation. Compare every change with its exact
parent and inspect segment support before interpreting deltas.

## Starting configuration and expected effects
No metric reweighting is permitted. A hypothesis should predict GAUC and nDCG@5 separately because
loss and sampling changes can trade them off.

## Diagnostics and risks
Validation-selected classification thresholds are optimistic. Score scaling can change thresholded
metrics without changing rank; zero-positive users make macro top-k metrics harsh but defined.

## Cheapest check and clean experiment
Run synthetic perfect, tied, all-negative, and non-finite cases before trusting evaluator changes.

## Related cards and sources
See `task.kuairand_ranking` and `experiment.scarce_budget`.

