# Starter Kit empirical priors for experiment selection

## Summary and mechanism
The KuaiRand Starter Kit reports two committee-tested directions that did not produce a meaningful
gain in its baseline ablations. Adding a broad bundle of static metadata produced primary 0.5940
versus 0.5950 with the five-domain setup, while changing the embedding dimension across 8, 16, and
32 produced 0.5895, 0.5902, and 0.5887. Treat these results as empirical priors against spending a
scarce experiment on an undifferentiated static-feature bundle or a capacity-only increase.

The same guidance identifies more promising unexplored directions: ranking-aligned pairwise or
listwise losses; causal user-history sequence modeling; auxiliary-task learning; censored watch-time
modeling; temporal features and distribution shift; and a separate random-exposure robustness audit.
Architecture changes such as DeepFM, DCN, or xDeepFM come after objectives, history, multitask, and
watch-time modeling unless current measured evidence gives a specific reason to change that order.

## When to use / avoid
Use this card when generating, comparing, or selecting hypotheses. Deprioritize a proposal whose
only mechanism is adding many static fields or increasing embedding width, depth, or parameter
count. Require such a proposal to cite current train-only screening or scored-experiment evidence
that identifies a specific missing interaction, segment weakness, or underfitting signature.

Do not turn these priors into hard bans. The Starter Kit did not test every feature interaction,
causal aggregate, sequence representation, or architecture. A narrowly targeted item/context
feature or interaction-aware feature can still be worthwhile when current evidence supports it.
Likewise, a new sequence, multitask, or ranking architecture adds a different mechanism rather than
merely scaling the existing model.

## Requirements and implementation
Pure user-side first-order terms are constant within a user's candidate group and therefore cannot
change within-user ranking. User features must interact with an item, author, context, or causal
history signal to affect GAUC or nDCG@5. Static item or context fields should be added selectively,
with missingness handling and a declared within-user variation mechanism, rather than as an
all-fields bundle.

When choosing among otherwise credible candidates, use this default priority order:

1. Ranking-aligned loss or sampling changes.
2. Causally truncated user-history representations.
3. Auxiliary engagement tasks that support `long_view`.
4. Censored or otherwise task-faithful watch-time modeling.
5. Temporal/context features and distribution-shift handling.
6. Architecture changes that introduce a justified interaction mechanism.
7. Capacity-only or broad static-feature changes.

The random-exposure log may support a separately declared robustness audit, but it must not silently
replace the official split, objective, or validation population.

## Starting configuration and expected effects
Prefer one clean ablation that changes a single mechanism. For a ranking objective, retain the
current scorer and compare pointwise against pairwise or listwise training. For history modeling,
start with a bounded causal history and a compact candidate-aware pooling method. For multitask or
watch-time modeling, keep `long_view` as the official target and report separate GAUC and nDCG@5
effects. Do not claim a numerical gain from the Starter Kit findings; they rank research directions
but are not measurements of the current parent experiment.

## Diagnostics and risks
A capacity-only proposal is weak when training and validation metrics are both stable, added width
does not change score variance, or prior widths were indistinguishable. A static-feature proposal is
weak when the train-only screen is flat or negative, fields are user-constant, or gains disappear
after retaining ID interactions. Conversely, persistent underfitting, a specific cold-item/context
slice, or a positive causal feature screen can justify revisiting a targeted member of a generally
low-priority family.

The ranked exploration list is committee guidance, not official ParLLiaMent experiment evidence.
Never place this card's ID in `evidence_ids`; cite it only as literature or fixed research context.

## Cheapest check and clean experiment
Before using an official experiment, consult the cached train-only temporal screen and the scored
archive. Reject an unqualified capacity sweep or all-static-feature bundle. If revisiting either
family, predeclare the exact interaction or diagnostic it addresses and change only that feature or
capacity mechanism. Prefer the highest-ranked untested direction that fits the timeout and current
code contract.

## Related cards and sources
See `task.kuairand_ranking`, `task.experiment_protocol`, `task.leakage_policy`,
`objective.within_user_ranknet_pairwise_loss`, `objective.mixed_group_listnet_top1_loss`,
`features.causal_behavior_history_features`, `architecture.candidate_conditioned_history_attention`,
`objective.shared_trunk_auxiliary_behavior_multitask_loss`, and
`evaluation.random_exposure_generalization_audit`. Source: the repository-owned KuaiRand Starter
Kit README, especially “Where to Start Modifying,” and its `ablation_features.py` reproduction
script.
