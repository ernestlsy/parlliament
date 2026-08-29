# Recommender experiment knowledge base

## Fixed task facts

- The task is within-user ranking over logged KuaiRand-Pure impressions, using `long_view` as the
  binary relevance label.
- Validation primary is the mean of positive-count-weighted per-user AUC and mean nDCG@5. A
  user-only constant cannot alter either ranking.
- Experiment 0 is deliberately unscored and architecture-neutral. It is only a runnable user/item
  additive scaffold; infer performance and promising directions from counted experiment records.

## Diagnostic metric interpretation

- `primary`, GAUC, and nDCG@5 remain the official optimization and stopping metrics.
- Accuracy, precision, recall, F1, specificity, balanced accuracy, and Matthews correlation use a
  validation threshold chosen to maximize F1. They diagnose class-separation tradeoffs but are
  optimistically measured on the same split that selects their threshold.
- Global AUC and average precision summarize row-level discrimination; GAUC remains more aligned
  with the actual within-user task.
- Precision@5, Recall@5, MAP@5, MRR@5, and HitRate@5 reveal different top-of-list failure modes.
  Macro averages include zero-positive users as zero, consistent with the fixed nDCG convention.
- Score mean and spread help flag collapsed or numerically unstable prediction distributions. Do
  not reward arbitrary score scale changes because only ordering affects the official metrics.

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
5. Interaction-capable models such as FM, DeepFM, DCN, and xDeepFM are valid hypotheses when their
   expected benefit and resource cost are stated explicitly.
6. Time-of-day, date, recency, and train-to-validation drift may matter. Validate that temporal
   features are known at impression time.
7. The random-exposure log can diagnose selection-bias overfitting, but it must not change the fixed
   official validation score used for stopping.

## Experiment hygiene

- Prefer a single interpretable change per hypothesis. Keep unchanged roles inactive.
- Compare against the referenced parent's exact config and code, not only the global best.
- Reject label leakage, validation fitting, prediction artifacts containing NaN/Inf, and changes that
  optimize a user-level constant.
- Resource feasibility matters: the full dataset should not be subjected to an unbounded all-pairs
  or full-history computation.
