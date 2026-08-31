# Validation population, label support, and pair novelty

## Summary and mechanism
The official validation slice has 124,909 impressions from 22,377 users and 5,951 videos, with a
31.33% long-view rate. It is mostly warm at the entity level but nearly novel at the user-video-pair
level, which favors transferable user/item/context structure over pure pair memorization.

## When to use / avoid
Use these facts when choosing architectures, interpreting segments, or estimating the value of
metadata and history. Avoid calling the validation set a conventional cold-start set: only 422 users
and seven videos are unseen from training.

## Requirements and implementation
Training has 1,092,750 unique user-video pairs; validation has 121,337. Of validation's unique pairs,
119,363 (98.37%) do not occur in training. Validation users have a median of four and mean of 5.582
impressions, so nDCG@5 often covers most or all of a user's candidate list.

## Starting configuration and expected effects
Prioritize item, author, context, and historical-preference interactions that transfer across pairs.
Report warm/cold entity slices and activity slices, but optimize the fixed aggregate primary score.
Treat improvements for short candidate lists carefully because top-five metrics may saturate.

## Diagnostics and risks
Validation has 6,785 zero-positive users (30.32%), 2,663 all-positive users, and 12,929 mixed-label
users. Zero-positive users necessarily contribute zero nDCG and no GAUC evidence; all-positive users
contribute nDCG but no GAUC evidence. Thus GAUC is driven by the mixed-label population while nDCG
averages across all users. Metric movements may come from different subpopulations.

## Cheapest check and clean experiment
Before proposing a model, calculate within-user score variance and metrics by label-support bucket,
candidate-count bucket, warm/cold status, and pair-seen status. Prefer a change whose train-only
holdout gain is not confined to one tiny segment.

## Related cards and sources
See `task.kuairand_ranking`, `dataset.user_metadata`,
`dataset.video_metadata_and_statistics`, and `evaluation.within_user_metrics`. Counts were measured
directly from the supplied standard-exposure logs using Ernest's official date boundaries.
