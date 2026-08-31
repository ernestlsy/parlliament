# BCE plus LambdaRank-style nDCG@5 hybrid

## Summary and mechanism

Train the existing scorer with a broad, pointwise **user-normalized binary cross-entropy (BCE)** term plus a group-local, pairwise logistic ranking term. The added term considers positive–negative item pairs within the same user/candidate group and weights each pair by the absolute change in nDCG@5 that would result from swapping their current predicted ranks. This concentrates ranking pressure on misorders that affect the first five positions.

For a group \(g\), with logits \(s_i\), binary labels \(y_i\in\{0,1\}\), and a temperature \(\tau\), use:

\[
L = L_{\mathrm{BCE,user}} + \lambda L_{\mathrm{rank}}
\]

\[
L_{\mathrm{rank},g}=\frac{\sum_{(p,n)} w_{pn}\,\operatorname{softplus}(-(s_p-s_n)/\tau)}{\sum_{(p,n)}w_{pn}+\epsilon},
\]

where \(p\) is a positive, \(n\) is a negative, and \(w_{pn}=|\Delta\mathrm{nDCG@5}_{pn}|\). Compute ranks by sorting *only the candidates in that group* by current score. For binary labels, with \(D(r)=1/\log_2(r+1)\) for \(r\leq5\) and zero otherwise:

\[
w_{pn}=\frac{|D(r_p)-D(r_n)|}{\mathrm{IDCG@5}_g}.
\]

Use zero weight when \(\mathrm{IDCG@5}_g=0\), and exclude same-label pairs. Treat ranks, pair selection, and swap weights as detached from autograd; gradients flow through logits in the logistic pair loss, not through sorting.

This is a pragmatic hybrid, not a claim that the sum is the original LambdaRank objective. LambdaRank introduced the idea of scaling RankNet-style pairwise updates by the target metric change caused by swapping a pair; the original treatment targets NDCG with generally graded relevance labels. Binary-label nDCG@5 weighting is a direct specialization. ([papers.neurips.cc](https://papers.neurips.cc/paper/2971-learning-to-rank-with-nonsmooth-cost-functions.pdf?utm_source=openai))

## When to use / avoid

**Use when:**

- GAUC is acceptable but nDCG@5 lags, especially when inspection shows positive items just below the top five or negatives occupying top-five slots.
- Each training group reconstructs a real candidate slate and has enough mixed-label groups to create positive–negative pairs.
- BCE is already stable and calibrated enough that a small, metric-focused adjustment is preferable to replacing the parent objective.

**Avoid when:**

- Most groups contain one candidate, only one label class, or unreliable/incomplete candidate sets. The ranking component then has little usable signal or optimizes an artifact of candidate construction.
- Group membership, labels, or candidate availability may include information unavailable at scoring time.
- The relevant product metric is not top-five ranking, or a validation split cannot evaluate both GAUC and nDCG@5 under the same group definition.

## Requirements and implementation

1. **Reconstruct groups before batching.** Each example needs a long-view binary target and a within-user candidate-group ID. Group together only candidates jointly eligible at the ranking decision. Candidate position may be used to reconstruct the historical training slate, but do not use a post-ranking/display position as a predictive feature.
2. **Keep all loss operations group-local.** Never sort, pair, normalize, or compute IDCG across users/groups. A cross-group pair is a leakage and objective-definition bug.
3. **Use a stable pointwise base.** Compute BCE from logits with a numerically stable implementation. Compute per-group mean BCE, then average groups so very large candidate sets do not dominate merely through size.
4. **Compute pair weights from detached current scores.** Within each group, deterministically sort descending logits; use a stable secondary key only to resolve exact ties reproducibly. Calculate \(r_p\), \(r_n\), discounts, IDCG@5, and detached \(w_{pn}\).
5. **Enumerate only useful pairs.** For each positive-negative pair, skip it if \(w_{pn}=0\). With nDCG@5, swaps between two candidates both below rank five have zero direct weight. For very large groups, retain all candidates that can yield nonzero swap weight, or sample pairs with an explicit sampling scheme and monitor approximation error.
6. **Normalize the ranking term within group.** Divide by the group’s sum of nonzero pair weights, then average over valid mixed-label groups. This makes one fixed \(\lambda\) more portable across group-size distributions. Log the number of valid ranking groups and total nonzero-weight pairs.
7. **Handle degenerate groups explicitly.** A group with no positive, no negative, or zero IDCG contributes zero ranking loss. Decide separately whether it remains in BCE; normally it should, because pointwise supervision remains informative.

The RankNet paper provides the pairwise probabilistic/logistic ranking foundation, while the LambdaRank paper describes NDCG-swap-scaled RankNet-style updates. Primary identifiers: Burges et al., *Learning to Rank using Gradient Descent*, ICML 2005, DOI `10.1145/1102351.1102363`; Burges, Ragno, and Le, *Learning to Rank with Nonsmooth Cost Functions*, NeurIPS 2006. ([doi.org](https://doi.org/10.1145/1102351.1102363?utm_source=openai))

## Starting configuration and expected effects

Start from the converged BCE model and retain its optimizer, features, sampling, schedule, and early-stopping protocol.

- **Default:** \(\tau=1\), \(\lambda=0.1\), per-group BCE normalization, per-group pair-weight normalization, and all nonzero-weight positive–negative pairs.
- **Coefficient search:** test one fixed grid such as \(\lambda\in\{0,0.03,0.1,0.3,1.0\}\). If ranking-loss normalization differs from the specification above, recalibrate this grid rather than transferring coefficients blindly.
- **Temperature:** keep \(\tau=1\) first. Only test \(\{0.5,1,2\}\) after selecting a viable \(\lambda\); varying \(\lambda\) and \(\tau\) together confounds the first experiment.
- **Selection rule:** choose the smallest \(\lambda\) that improves validation nDCG@5 without violating a predeclared GAUC guardrail and without worsening key slice metrics.

Expected effects are empirical rather than guaranteed: the hybrid may improve top-five ordering because its nonzero pair weights emphasize swaps involving ranked positions 1–5. It can leave GAUC roughly unchanged, improve it, or reduce it if the ranking term overweights a narrow set of top-position conflicts relative to broad pointwise discrimination. Do not claim a fixed lift magnitude without a controlled evaluation on the target data.

## Diagnostics and risks

- **No nDCG@5 movement; ranking loss near zero:** mixed-label groups may be rare, groups may be reconstructed incorrectly, pairs may all sit below rank five, or IDCG/discount indexing may be wrong. Log valid-group rate, nonzero-pair rate, mean pair weight, and the ranks of both endpoints.
- **nDCG@5 falls immediately:** check the sign in `softplus(-(s_pos-s_neg)/tau)`, score sort direction, 1-indexed rank discounting, and whether labels identify the desired long-view positive outcome.
- **GAUC declines while nDCG@5 rises:** \(\lambda\) is probably too large, pair normalization may be unstable, or the top-five target is genuinely trading off against broad ordering. Reduce \(\lambda\) before changing architecture.
- **Large train/validation gap:** group-specific memorization, target leakage, or nonstationary candidate generation are likely. Split by the appropriate time/entity boundary and ensure every group’s candidates and labels obey the split cutoff.
- **Unstable training or loss spikes:** a few groups may have many pairs or unusually large gradients. Verify group normalization, clip gradients if clipping already belongs to the BCE baseline protocol, and inspect group-size tails.
- **High compute cost:** naive enumeration is \(O(P_gN_g)\) for \(P_g\) positives and \(N_g\) negatives per group, plus sorting. Measure group-size percentiles before rollout; cap or sample only with a documented, fixed policy.
- **Exposure or censoring risk:** observed non-click/non-conversion labels may reflect what was exposed rather than true negative preference. This card does not correct exposure bias; it only reweights supervised pair comparisons.

## Cheapest check and clean experiment

**Cheap train-only check:** run one forward pass over a fixed training shard with gradients disabled. For each group, independently recompute nDCG@5 before and after swapping a sampled positive-negative pair, and assert that the implementation’s \(|\Delta\mathrm{nDCG@5}|\) matches the direct calculation. Also assert: no pair crosses group IDs; all pair endpoints have opposite labels; zero-weight pairs have both current ranks above five; and a positive ranked below a negative yields a positive corrective pair-loss gradient with respect to \(s_p-s_n\).

**Clean single-variable experiment:** clone the BCE parent checkpoint and run matched training jobs differing only in \(\lambda\): `0` versus the prespecified coefficient grid. Freeze data split, candidate construction, initialization/checkpoint, negative sampling, optimizer, learning-rate schedule, batch formation, seed set, stopping rule, and evaluation code. Report validation and held-out GAUC plus nDCG@5 overall, by group-size bucket, by number-of-positives bucket, and by time slice. Select on validation only; run the held-out test once for the selected coefficient.

## Related cards and sources

**Related cards:** `objective.user_normalized_binary_cross_entropy`, `objective.within_user_ranknet_pairwise_loss`, `evaluation.within_user_metrics`, `task.leakage_policy`, `task.experiment_protocol`, `dataset.inventory_and_splits`, `task.prediction_artifact`.

**Primary sources:**

- Burges, Shaked, Renshaw, Lazier, Deeds, Hamilton, and Hullender (2005), *Learning to Rank using Gradient Descent*, ICML. DOI: `10.1145/1102351.1102363`. ([doi.org](https://doi.org/10.1145/1102351.1102363?utm_source=openai))
- Burges, Ragno, and Le (2006), *Learning to Rank with Nonsmooth Cost Functions*, NeurIPS. Proceedings PDF: `https://papers.n