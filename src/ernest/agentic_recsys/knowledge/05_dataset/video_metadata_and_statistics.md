# Video metadata and aggregate-statistics tables

## Summary and mechanism
Both video tables contain exactly one row for each of 7,583 videos. The basic table provides
`author_id`, `video_type`, `upload_dt`, `upload_type`, visibility, duration, dimensions, `music_id`,
`music_type`, and `tag`. The statistics table provides many exposure, play, completion, engagement,
and negative-feedback aggregates.

## When to use / avoid
Use basic metadata for generalization to unseen user-video pairs and the upload date for age features
when it precedes the impression. Avoid using aggregate statistics until their snapshot time and
observation window are proven to precede every scored impression.

## Requirements and implementation
Join by `video_id`, preserving impression row order and using explicit unknown/missing values. The
basic table has 239 missing `video_duration` values, 203 missing `music_type` values, and 96 missing
`tag` values. The supplied statistics table has no empty cells, but completeness does not establish
causal availability. Fit numeric scaling and categorical vocabularies on training only.

## Starting configuration and expected effects
Start with author, type, upload type, tag, music, log-duration, aspect-ratio, and causal video age.
Cross these with user or context signals so the model can alter within-user ordering. If aggregate
statistics pass provenance review, prefer log transforms, rate denominators, clipping, and a small
well-documented group rather than all columns at once.

## Diagnostics and risks
Training observes 7,538 videos; validation observes 5,951 with seven unseen versus training; test
observes 5,982 with nine unseen. Entity-level item cold start is rare, but pair novelty is high.
Snapshot aggregates may encode future validation/test behavior and are therefore quarantined by
default. Raw counts can also proxy exposure policy rather than relevance.

## Cheapest check and clean experiment
First screen basic metadata groups on the internal temporal holdout. For every statistics column,
locate generation timestamps or rebuild it from strictly prior training events; otherwise exclude it.
Test one coherent metadata group while keeping the objective and training schedule fixed.

## Related cards and sources
See `task.leakage_policy`, `dataset.population_and_pair_shift`, and `dataset.random_exposure_log`.
Schema, coverage, and missingness were measured from the supplied CSVs.
