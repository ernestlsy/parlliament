# User metadata table

## Summary and mechanism
`user_features_pure.csv` has one row for each of 27,285 users. It contains activity status; low-
activity, live-streamer, and author flags; follow/fan/friend counts and range buckets; registration
age and bucket; and 18 anonymized `onehot_feat*` fields.

## When to use / avoid
Use these fields for cold-user fallback, regularized user-item/context interactions, and segment
analysis. Avoid expecting a user-only additive term to improve within-user ranking because it is
constant over all candidate impressions for that user.

## Requirements and implementation
Fit category vocabularies and numeric transforms on training users only, retain an unknown value,
and add explicit missing indicators where useful. The supplied table has 874 missing values in
`onehot_feat4` and 714 each in `onehot_feat12` through `onehot_feat17`; other columns have no empty
CSV cells. Do not assume anonymized values are ordinal merely because they parse as numbers.

## Starting configuration and expected effects
Start with coarse categorical embeddings or hashed crosses between user metadata and item author,
tag, type, duration, or request tab. Keep embedding dimensions small and regularized. Expect any
gain to appear through heterogeneous preferences, not through standalone user bias.

## Diagnostics and risks
Training contains 26,210 users. Validation contains 22,377 users, only 422 of whom are unseen in
training; test contains 23,875 users with 777 unseen. Aggregate gains can therefore hide weak true
cold-user performance. High-cardinality anonymized fields can overfit or duplicate user ID.

## Cheapest check and clean experiment
Use the train-only feature screen to compare metadata groups and their interactions, then inspect
warm/cold-user segments. A clean official experiment adds only the best supported interaction group
with fixed model capacity elsewhere.

## Related cards and sources
See `dataset.population_and_pair_shift`, `dataset.video_metadata_and_statistics`, and
`task.kuairand_ranking`. Schema, coverage, and missingness were measured from the supplied CSVs.
