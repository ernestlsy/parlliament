# Impression-log fields and availability classes

## Summary and mechanism
Every impression log has `user_id`, `video_id`, `date`, `hourmin`, `time_ms`, `is_click`, `is_like`,
`is_follow`, `is_comment`, `is_forward`, `is_hate`, `long_view`, `play_time_ms`, `duration_ms`,
`profile_stay_time`, `comment_stay_time`, `is_profile_enter`, `is_rand`, and `tab`.

## When to use / avoid
Use identifiers, impression time, request context, and known item duration as candidate inputs. Use
engagement outcomes as training labels or as prior-event history only. Never use the current row's
engagement, watch time, or `long_view` to score that same impression.

## Requirements and implementation
Treat `is_click`, `is_like`, `is_follow`, `is_comment`, `is_forward`, `is_hate`, `long_view`,
`play_time_ms`, `profile_stay_time`, `comment_stay_time`, and `is_profile_enter` as post-impression
outcomes. `date`, `hourmin`, `time_ms`, `tab`, and `is_rand` describe time or exposure context;
`duration_ms` describes the candidate video. Historical features must be emitted before updating
state with the current outcome.

## Starting configuration and expected effects
Start with categorical IDs/context, cyclic or bucketed time, log-scaled duration, and causally
lagged counts/rates. Auxiliary engagement targets are highly imbalanced in standard training:
measured positive rates are 46.34% click, 33.66% long-view, 1.87% like, 0.257% comment, 0.101%
follow, and 0.100% forward. Weight or sample auxiliary tasks deliberately.

## Diagnostics and risks
The logs have no empty CSV cells in this supplied snapshot, but valid zero values are common and
must not be interpreted as missing. `hourmin` is HHMM-like rather than elapsed minutes. Outcome
columns can produce spectacular but invalid gains if accidentally admitted as contemporaneous
features.

## Cheapest check and clean experiment
Maintain an explicit allowlist and denylist, perturb each current-row outcome, and assert engineered
features for that row remain unchanged. For histories, compare a row-by-row causal implementation
against a small hand-calculated sequence.

## Related cards and sources
See `task.leakage_policy`, `dataset.inventory_and_splits`, and
`dataset.random_exposure_log`. Field names and rates were measured from the supplied CSVs.
