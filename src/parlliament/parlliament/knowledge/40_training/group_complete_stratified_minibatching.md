# Group-complete, label-stratified minibatching

## Summary and mechanism

For pairwise and listwise learning-to-rank (LTR), treat one displayed candidate group—typically all candidates shown for one user request, slate, or impression—as the atomic sampling unit. Do **not** independently sample candidate rows and reconstruct partial groups afterward. A RankNet-style loss is defined over preference pairs, while listwise methods such as ListNet take a candidate list as the training instance; dropping candidates changes the comparisons and list distribution presented to the loss. ([icml.cc](https://icml.cc/2015/wp-content/uploads/2015/06/icml_ranking.pdf?utm_source=openai))

Partition train-only groups by their long-view labels:

- **Zero-positive:** all displayed candidates have label 0.
- **Mixed-label:** at least one positive and at least one negative.
- **All-positive:** all displayed candidates have label 1.

Sample groups from these strata, then emit every row in each selected group. Mixed groups supply valid positive-vs-negative pairs and non-degenerate listwise targets. Zero- and all-positive groups have no binary within-group ordering signal, but retaining some prevents the model from training only on unusually engaged slates and lets any pointwise auxiliary term see the broader exposure distribution.

This is a sampling and optimization intervention, not a change to labels, candidate eligibility, evaluation protocol, or loss definition. It assumes the logged group corresponds to the actual candidate set whose order the model is meant to improve.

## When to use / avoid

**Use when** training `objective.within_user_ranknet_pairwise_loss`, `objective.bce_lambdarank_ndcg5_hybrid`, or `objective.mixed_group_listnet_top1_loss`, especially when GAUC and nDCG@5 are computed within candidate groups and ordinary row sampling yields batches with few or no mixed groups.

**Avoid or defer when** the model is strictly pointwise and group reconstruction severely reduces throughput, or when the group identifier is ambiguous, incomplete, or merges unrelated exposures. A wrong group boundary is worse than a less sophisticated sampler: it creates artificial comparisons and can contaminate a listwise target.

## Requirements and implementation

Required fields are: a stable user candidate-group identifier, the long-view label, and either displayed position or a stable candidate/impression identifier for deterministic ordering and joins. Build the strata **after** applying the train split and any train-prefix feature construction; never use validation/test labels, future interactions, or post-hoc candidates to classify a training group.

Implement the sampler as follows:

1. Materialize a train-only table keyed by `(group_id, candidate_id)` and assert each key is unique.
2. Aggregate each group into `n_candidates`, `n_positive`, label class, and optional logging metadata such as surface, time bucket, and source.
3. Maintain three shuffled queues of group IDs. Sample a target number of groups from each queue, fetch all corresponding candidate rows, and pack whole groups until a row/token budget is reached.
4. Use padding plus a group mask for dense listwise implementations, or concatenate groups with segment offsets for pairwise implementations. Never form pairs across group boundaries.
5. Compute pairwise loss only on unequal-label pairs. Normalize by valid-pair count per group, then average over contributing groups; otherwise large groups can dominate simply because they contain quadratically more pairs. The original LTR literature notes that pair construction can vary substantially by query and can bias training toward groups producing more pairs. ([microsoft.com](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/tr-2007-40.pdf?file=tr-2007-40.pdf&utm_source=openai))
6. Log the realized—not just requested—group mixture, candidate count distribution, valid-pair count, padded-row fraction, and fraction of batches with at least one mixed group.

Use deterministic group hashing for distributed assignment and checkpoint the sampler RNG state. Keep all rows of a group on one worker; splitting a group across workers silently removes local comparisons unless gradients and group state are explicitly synchronized.

## Starting configuration and expected effects

Start with a **group-count** mixture of 60% mixed, 20% zero-positive, and 20% all-positive groups. If all-positive groups are rare, allocate their unused share to zero-positive groups; do not oversample with replacement so aggressively that a small set of mixed groups appears repeatedly within an epoch. Tune the mixed-group share over 40–80%, while preserving at least 10% representation for each available homogeneous stratum. Sample by group count, not row count.

Set a batch budget based on total candidates, for example 512–4,096 candidate rows per optimizer step, with a cap of 20–100 candidates per group. For oversized groups, prefer a documented deterministic truncation rule based only on information available at serving time, or exclude them consistently from both the relevant training comparison and its diagnostic. Randomly dropping negatives from only training groups changes the task; if used as a compute approximation, make it a separately evaluated treatment.

Expected effects are empirical, not guaranteed: when mixed groups are scarce, the sampler should increase the frequency of informative within-group gradients and may improve GAUC and nDCG@5. It can also reduce apparent gains if the original row sampler's implicit weighting toward large or high-activity groups was beneficial for the deployment population. Do not claim an improvement without a fixed-split experiment.

## Diagnostics and risks

**Data leakage:** deriving group membership from a later log join, using long-view outcomes outside the label horizon, or building groups from a post-ranking candidate inventory leaks future information. Validate that every candidate timestamp, feature cutoff, and label window obeys the train split and leakage policy.

**Compute risk:** group-complete batches have variable size. Diagnostic signature: high padding, out-of-memory events concentrated in a few large groups, or unstable step time. Remedies are a candidate-row budget, length bucketing, and a maximum group-size policy recorded in experiment metadata.

**Sampling distortion:** excessive mixed-group oversampling changes the training distribution. Diagnostic signature: pairwise/listwise training loss improves while held-out BCE calibration worsens, or gains appear only on mixed groups but not on the full evaluation population. Reduce the mixed share, retain homogeneous groups, or add/reweight a pointwise term deliberately.

**Broken grouping:** diagnostic signature: unusually large groups, duplicate candidates, impossible timestamp spans, or large disagreement between logged order and reconstructed order. Stop and repair the key; do not mask the issue through random subsampling.

**Objective mismatch:** zero-positive and all-positive binary groups yield no unequal-label RankNet pairs. If they dominate a pairwise-only batch, valid-pair count collapses and gradients become sparse. Either ensure each batch contains mixed groups or combine the ranking loss with a clearly specified pointwise objective.

## Cheapest check and clean experiment

**Cheapest train-only check:** before training, compare the existing row sampler with the proposed sampler over 1,000 simulated batches. Report: (1) percentage of batches containing a mixed group, (2) valid unequal-label pairs per step, (3) distinct groups and rows per step, (4) candidate-count and label-class mixture, and (5) padding/oversize rejection rate. This requires no validation labels and directly tests whether the intervention restores ranking signal.

**Clean experiment:** keep model, features, optimizer, total candidate-row budget, number of optimizer steps, random seeds, split, label horizon, loss weights, and evaluation code fixed. Change only the minibatch constructor: baseline independent-row sampling versus group-complete stratified sampling. Evaluate GAUC and nDCG@5 on the identical held-out groups, and report confidence intervals across seeds plus stratified results for zero-positive, mixed-label, and all-positive groups. Also report throughput and valid-pair counts, so a metric change is interpretable as an optimization effect rather than extra compute.

## Related cards and sources

Related cards: `dataset.interaction_log_schema`, `dataset.inventory_and_splits`, `evaluation.within_user_metrics`, `task.leakage_policy`, `task.experiment_protocol`, `objective.within_user_ranknet_pairwise_loss`, `objective.bce_lambdarank_ndcg5_hybrid`, `objective.mixed_group_listnet_top1_loss`, `objective.user_normalized_binary_cross_entropy`.

Primary sources: Burges et al., *Learning to Rank using Gradient Descent* (RankNet), DOI: `10.1145/1102351.1102363`. ([microsoft.com](https://www.microsoft.com/en-us/research/publication/learning-to-rank-using-gradient-descent/?msockid=0993ece6eaa36d5b2babfa44eb816ca2&utm_source=openai)) Cao et al., *Learning to Rank: From Pairwise Approach to Listwise Approach* (ListNet), DOI: `10.1145/1273496.1273513`. ([microsoft.com](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/tr-2007-40.pdf?file=tr-2007-40.pdf&utm_source=openai)) Xia et al., *Listwise Approach to Learning to Rank: Theory and Algorithm*, DOI: `10.1145/1390156.1390306`. ([mlanthology.org](https://mlanthology.org/icml/2008/xia2008icml-listwise/?utm_source=openai))

### Audited web sources

- Learning to Rank using Gradient Descent: <https://icml.cc/2015/wp-content/uploads/2015/06/icml_ranking.pdf?utm_source=openai>
- Learning to Rank: From Pairwise Approach to Listwise Approach: <https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/tr-2007-40.pdf?file=tr-2007-40.pdf&utm_source=openai>
- Learning to Rank using Gradient Descent - Micr