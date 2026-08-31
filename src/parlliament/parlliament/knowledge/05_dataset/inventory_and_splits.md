# Supplied KuaiRand-Pure files and official splits

## Summary and mechanism
The supplied data directory has three 19-column impression logs and three entity tables. ParLLiaMent's
official temporal split uses standard-exposure rows: train 2022-04-08 through 2022-04-21, validation
2022-04-22 through 2022-04-28, and test 2022-04-29 through 2022-05-08.

## When to use / avoid
Use this inventory before choosing data sources or defining temporal features. Avoid treating file
boundaries as split boundaries: the later standard log contains both validation and test dates.
Avoid treating the random-exposure file as official validation or test data.

## Requirements and implementation
Measured directly from the supplied CSVs: `log_standard_4_08_to_4_21_pure.csv` has 1,141,112 rows
covering 2022-04-09 through 2022-04-21; `log_standard_4_22_to_5_08_pure.csv` has 295,497 rows over 17
dates. Date filtering yields 124,909 validation rows and 170,588 test rows. The random log has
1,186,059 rows over 2022-04-22 through 2022-05-08. Entity tables contain 27,285 users and 7,583
videos, with one row per ID.

## Starting configuration and expected effects
Read standard logs in source order, select rows by integer `date`, and preserve that order for
validation inference. Fit encoders and transforms on training only. Use explicit unknown buckets for
entities or values absent from training.

## Diagnostics and risks
The nominal train range includes April 8, but this supplied training file's observed minimum date is
April 9. Hard-coding expected daily row counts or concatenating all standard rows before fitting can
silently leak validation/test information. File names alone do not establish feature availability.

## Cheapest check and clean experiment
Before a run, assert filenames, headers, row counts, observed date bounds, unique IDs, and the exact
124,909-row validation selection. Fingerprint inputs so cached screening is invalidated after a data
change.

## Related cards and sources
See `dataset.interaction_log_schema`, `dataset.user_metadata`,
`dataset.video_metadata_and_statistics`, `dataset.random_exposure_log`, and
`task.prediction_artifact`. Counts were measured from the supplied KuaiRand-Pure CSVs; split rules
come from ParLLiaMent's fixed evaluator.
