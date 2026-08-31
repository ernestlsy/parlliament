# Candidate-group shared representation scoring

## Summary and mechanism
For an impression group \(g\) containing candidates \(i=1,\ldots,n_g\), split the scorer into a candidate-invariant state and a candidate-specific head:

\[
h_g = E_{user}(u_g,\; H_{<t_g},\; c_g), \qquad s_{g,i}=F(h_g, v_{g,i}, x_{g,i}).
\]

Compute \(h_g\)—user embeddings, strictly pre-cutoff history encoding, metadata transforms, and group-level context—once, then broadcast it over the group’s candidate-video tensors and execute one vectorized score call. This is an inference refactor, not a modeling change: with identical inputs and the same computational graph per candidate, it preserves the model’s mathematical scores and ranking. Ranking systems commonly score a reduced candidate list using shared user/context and item/impression features, so repeated row-wise evaluation is avoidable work when those inputs are truly group-invariant. ([arxiv.org](https://arxiv.org/pdf/1606.07792?utm_source=openai))

Use `[G, L, ...]` padded batches plus a candidate mask, or concatenate candidates and retain `group_id`/offsets for reconstruction. Broadcasting supplies compatible singleton dimensions without materializing copies under standard tensor semantics. ([docs.pytorch.org](https://docs.pytorch.org/docs/stable/notes/broadcasting.html?utm_source=openai))

## When to use / avoid
**Use** when every candidate in a displayed group has the same user, request context, history cutoff, feature snapshot, and encoder mode; especially when causal attention or a sequence transformer dominates scoring cost.

**Avoid** when candidate position, a candidate-dependent history filter, sequential within-slate state, per-row timestamps, or candidate-conditioned attention changes the supposedly shared encoder input. Also avoid calling the result “bitwise identical” until it is tested on the deployed hardware and precision: batched and sliced operations can be mathematically equivalent yet produce slightly different floating-point results. ([docs.pytorch.org](https://docs.pytorch.org/docs/main/notes/numerical_accuracy.html?utm_source=openai))

## Requirements and implementation
1. Define an immutable `displayed_candidate_group_id`, canonical candidate order, request timestamp \(t_g\), and candidate offsets before feature construction.
2. Build history from events strictly before \(t_g\); cache or compute `history_state[g]` only after this cutoff is applied. Never construct it from all rows in the group or from post-impression interactions.
3. Factor the existing model explicitly into `encode_shared(group)` and `score_candidates(shared, candidate_features)`. Keep dropout disabled, normalization in inference mode, and all feature transforms/version lookups unchanged.
4. Form `shared: [G,D]`, candidates: `[G,L,D_v]`, and mask: `[G,L]`; use `shared[:,None,:].expand(-1,L,-1)` only at the head boundary. Score valid entries, then scatter outputs back by saved offsets to canonical row order.
5. Default to group-complete microbatches, sorted or bucketed by candidate count. Start with `G=32–256` groups per device batch; tune by p95 latency, padding fraction, and memory. If padding waste exceeds roughly 20–30%, bucket lengths or use packed candidate rows with a group-index gather.
6. Keep the old row-wise path as an oracle until parity and serving observability are established.

## Starting configuration and expected effects
Start with the trained pointwise scorer unchanged; share only computations proven independent of candidate identity. Common safe initial boundary: user-ID embedding, user metadata MLP, request-level numeric transforms, and causal history encoder. Keep all video/item embeddings, item statistics, candidate cross features, and candidate-conditioned attention in the per-candidate head.

Expected quality effect is **none by design**: GAUC and nDCG@5 should match the canonical scorer within the predeclared numerical tolerance. Any repeatable metric movement signals altered inputs, ordering, masking, nondeterminism, or an accidental architecture change—not an efficiency gain. The likely benefit is lower repeated encoder work and better accelerator utilization; actual latency and throughput gains depend on group size, encoder cost, padding, memory bandwidth, and kernel selection, so measure rather than assume a magnitude. Batched matrix operations and broadcasted matrix products are supported directly by common tensor runtimes. ([docs.pytorch.org](https://docs.pytorch.org/docs/stable/generated/torch.baddbmm.html?utm_source=openai))

## Diagnostics and risks
- **History leakage:** online/offline GAUC rises unexpectedly, especially for short-horizon labels; audit maximum event time in each encoded history against \(t_g\).
- **Cross-group contamination:** scores change when groups are repartitioned or shuffled; assert group IDs, offsets, and history-cache keys.
- **Order corruption:** nDCG@5 falls while row-level score comparisons look plausible; round-trip candidate IDs and verify exact canonical order before metric computation.
- **Mask/padding bug:** dummy candidates receive finite scores or affect top-k; require `mask=False` entries to be excluded before ranking.
- **Candidate-dependent “shared” state:** differences concentrate on groups with heterogeneous timestamps/positions or candidate-conditioned attention; move that computation back into the head.
- **Numerical drift:** small score deltas appear only under mixed precision or a changed batch shape. Treat semantic equivalence and bitwise equivalence separately; validate score error, rank agreement, and top-5 agreement. PyTorch documents that batched versus sliced computation is not guaranteed bitwise identical. ([docs.pytorch.org](https://docs.pytorch.org/docs/main/notes/numerical_accuracy.html?utm_source=openai))
- **Memory regression:** GPU utilization rises but p95 latency worsens; inspect padding, activation/workspace size, and tail group lengths.

## Cheapest check and clean experiment
**Cheap train-only check:** select several thousand training impressions without using labels in the computation. Run frozen row-wise and grouped inference with identical feature snapshots. Join by immutable row ID and require: identical candidate count and order per group; no history event at or after \(t_g\); finite scores only for valid candidates; maximum absolute score difference below a predeclared precision-specific tolerance; near-perfect rank agreement; and exact agreement of top-5 candidate IDs where ties are resolved by the same canonical rule. Investigate every non-tie top-5 mismatch.

**Clean experiment:** hold model checkpoint, data split, feature snapshots, hardware, precision, batch budget, and candidate groups fixed. Change only execution mode: row-wise baseline versus shared-state grouped scoring. Report p50/p95 latency per impression, impressions/s, accelerator memory, encoder calls per candidate group, score-delta distribution, GAUC, nDCG@5, top-5 agreement, and failure counts. Roll out only if quality parity and ordering integrity pass before claiming the optimization is exact.

## Related cards and sources
Related: `dataset.interaction_log_schema`, `dataset.inventory_and_splits`, `task.leakage_policy`, `features.causal_behavior_history_features`, `architecture.candidate_conditioned_history_attention`, `architecture.causal_sequence_transformer_ranker`, `training.group_complete_stratified_minibatching`, `evaluation.frozen_candidate_group_integrity_audit`, `evaluation.within_user_metrics`, `task.prediction_artifact`, `efficiency.dot_product_linear_residual_ranker`.

Primary sources: Cheng et al., *Wide & Deep Learning for Recommender Systems* (2016), for the retrieval-then-ranking setting and shared user/context versus impression features. ([arxiv.org](https://arxiv.org/pdf/1606.07792?utm_source=openai)) PyTorch documentation for broadcasting and batched matrix multiplication, plus its numerical-accuracy note on non-bitwise-equivalent batched execution. ([docs.pytorch.org](https://docs.pytorch.org/docs/stable/generated/torch.baddbmm.html?utm_source=openai))

### Audited web sources

- Wide & Deep Learning for Recommender Systems: <https://arxiv.org/pdf/1606.07792?utm_source=openai>
- Broadcasting semantics — PyTorch 2.13 documentation: <https://docs.pytorch.org/docs/stable/notes/broadcasting.html?utm_source=openai>
- Numerical accuracy — PyTorch main documentation: <https://docs.pytorch.org/docs/main/notes/numerical_accuracy.html?utm_source=openai>
- torch.baddbmm — PyTorch 2.13 documentation: <https://docs.pytorch.org/docs/stable/generated/torch.baddbmm.html?utm_source=openai>
