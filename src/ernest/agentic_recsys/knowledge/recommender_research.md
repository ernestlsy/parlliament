# Recommender experiment knowledge base

## Fixed task facts

- The task is within-user ranking over logged KuaiRand-Pure impressions, using `long_view` as the
  binary relevance label.
- Validation primary is the mean of positive-count-weighted per-user AUC and mean nDCG@5. A
  user-only constant cannot alter either ranking.
- The official five-field factorization-machine baseline uses user, video, author, tab, and duration
  bucket fields. Its published validation primary is 0.6016 with seed standard deviation around
  0.0008.
- Static feature expansion and larger embedding dimensions were already tested without meaningful
  gain. Treat materially equivalent ideas as low novelty.

## High-value hypothesis families

1. Pairwise BPR and listwise user-group objectives align training more directly with ranking than
   pointwise binary cross entropy. Pair construction must remain within user and avoid users lacking
   either class.
2. Sequential user histories (for example attention over recent positive and negative interactions)
   add behavioral information missing from static ID interactions. Prevent validation/test leakage by
   constructing every history strictly from events available before the target timestamp.
3. Auxiliary engagement objectives (`is_click`, `is_like`, `is_follow`, `is_comment`, `is_forward`)
   may regularize the long-view representation. The long-view output remains the scored head.
4. Watch-time regression is censored at video completion; one-sided or censored losses are more
   appropriate than ordinary squared error. This can be an auxiliary representation signal.
5. DeepFM, DCN, and xDeepFM are lower priority unless paired with a better objective or a meaningful
   interaction feature, because raw capacity was not the observed bottleneck.
6. Time-of-day, date, recency, and train-to-validation drift may matter. Validate that temporal
   features are known at impression time.
7. The random-exposure log can diagnose selection-bias overfitting, but it must not change the fixed
   official validation score used for stopping.

## Experiment hygiene

- Prefer a single interpretable change per hypothesis. Keep unchanged roles inactive.
- Compare against the referenced parent's exact config and code, not only the global best.
- Reject label leakage, validation fitting, prediction artifacts containing NaN/Inf, and changes that
  optimize a user-level constant.
- Resource feasibility matters: a 1.14-million-row NumPy baseline should not be replaced by an
  unbounded all-pairs or full-history computation.
