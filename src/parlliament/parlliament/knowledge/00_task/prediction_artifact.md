# Canonical validation prediction artifact

## Summary and mechanism
Each experiment must write `predictions_valid.npz`. The immutable evaluator independently reloads
canonical validation users and `long_view` labels, then associates them with scores by row position.
Generated experiment code cannot redefine the evaluation population or labels.

## When to use / avoid
Use this card when changing data loading, batching, filtering, sampling, or inference. Avoid sorting,
deduplicating, dropping, or shuffling validation rows unless inference restores exact canonical order.

## Requirements and implementation
The NPZ must contain exactly two one-dimensional arrays: integer `row_ids` and numeric `scores`.
Both arrays must have exactly the canonical validation length. `row_ids` must equal consecutive
integers from zero, and every score must be finite. Scores may be logits or arbitrary ranking values;
they need not be calibrated probabilities.

## Starting configuration and expected effects
Assign row IDs before any batching, carry them through inference, and sort outputs by row ID before
saving. Assert array shape, finiteness, uniqueness, and full coverage locally. Artifact correctness
does not improve metrics, but it is a prerequisite for an attempt to become a counted experiment.

## Diagnostics and risks
Common failures are filtered validation rows, order changes after a merge, duplicate row IDs, NaNs,
and writing extra arrays. A plausible score distribution cannot reveal misalignment; only explicit
row-order checks can. Validation threshold diagnostics do not change the raw scores used for ranking.

## Cheapest check and clean experiment
Run ParLLiaMent's contract check on a small slice, then load the finished NPZ with `allow_pickle=False`
and verify its exact keys, dimensions, integer row sequence, length, and finite scores before scoring.

## Related cards and sources
See `task.kuairand_ranking` and `dataset.inventory_and_splits`. The authoritative implementation is
`parlliament/evaluation.py::score_prediction_artifact`.
