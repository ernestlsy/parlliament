# Impression-time availability and leakage policy

## Summary and mechanism
A feature is eligible only when its value would be known before ranking the target impression.
Historical aggregates must exclude the current label and all future events.

## When to use / avoid
Use timestamps, request context, and causally accumulated histories. Avoid current-impression
outcomes, official-validation-derived fitting, future histories, and snapshots lacking availability
timestamps.

## Requirements and implementation
Sort training events by time, compute features from prior state, then update state after emitting the
row. Fit vocabularies, thresholds, and transforms on training data only. Treat engagement columns
such as `is_click`, `play_time_ms`, and `long_view` as post-impression outcomes for the current row.
Treat the video-statistics snapshot as unavailable until its observation window is established.

## Starting configuration and expected effects
Default unknown values explicitly and monitor coverage. Leakage can create implausibly large gains
in both GAUC and nDCG@5 that disappear under a temporal split.

## Diagnostics and risks
Warning signs are near-perfect metrics, cold rows outperforming warm rows unexpectedly, or gains
concentrated in post-event fields.

## Cheapest check and clean experiment
Perturb the current label and other current outcomes and verify the row's engineered features do not
change. Run the internal temporal holdout before an official experiment.

## Related cards and sources
See `dataset.interaction_log_schema`, `dataset.video_metadata_and_statistics`,
`features.causal_histories`, and `robustness.temporal_drift`.
