# KuaiRand within-user long-view ranking

## Summary and mechanism
The task scores the already logged impressions for each user; it is not full-catalog retrieval.
`long_view` is binary relevance. The official `primary` score is
`(GAUC + nDCG@5) / 2`, and it is the only experiment-selection and convergence objective.

## When to use / avoid
Use this card to reject changes that improve row calibration without changing within-user order.
Avoid treating the unscored seed or a user-only constant as a performance baseline.

## Requirements and implementation
Predictions must preserve canonical validation row order and contain one finite score per row. A
useful model must vary scores among a user's candidate items. GAUC is the positive-count-weighted
mean of per-user AUC over users containing both classes. nDCG@5 is macro-averaged over every user;
users with no positive impression receive zero nDCG.

## Starting configuration and expected effects
Always report GAUC, nDCG@5, and primary separately. Optimize ordering, not a validation-selected
classification threshold. Classification and supplementary ranking metrics are diagnostic only.

## Diagnostics and risks
Global AUC can rise while GAUC stalls. Zero-positive users contribute zero nDCG but do not provide
within-user positive-negative AUC evidence. User-only constants cannot change within-user order;
user features need to interact with an item, author, context, or history signal to affect the score.

## Cheapest check and clean experiment
Before training, verify score variance within user and the exact prediction artifact contract. Test
one ranking-relevant change against a scored parent.

## Related cards and sources
See `task.experiment_protocol`, `task.prediction_artifact`,
`dataset.population_and_pair_shift`, and `evaluation.within_user_metrics`. Task details are fixed by
Ernest's immutable evaluator; dataset context is Gao et al., “KuaiRand: An Unbiased Sequential
Recommendation Dataset with Randomly Exposed Videos” (2022).
