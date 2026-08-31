# Causal self-attentive sequence encoder ranker

## Summary and mechanism
Build a per-impression representation from the user’s **strictly pre-impression** sequence of watched/interacted video IDs. Embed the most recent capped history, add a positional or recency encoding, pass it through a small Transformer encoder with a strict causal attention mask, and take the final valid hidden state as the sequence representation. Concatenate it with user-ID, candidate-video, and impression-context embeddings; a shallow MLP produces the ranking logit.

This is a SASRec-style adaptation: causal self-attention lets the model weight earlier actions differently for each sequence position while preserving the rule that a position cannot attend to later events. Kang and McAuley introduced SASRec for next-item prediction, arguing that self-attention can capture longer-range behavior while focusing on a small relevant subset of prior actions; their reported experiments also show that the method can adapt across sparse and denser sequential datasets. ([cseweb.ucsd.edu](https://cseweb.ucsd.edu/~jmcauley/pdfs/icdm18.pdf?utm_source=openai))

The essential assumption is not merely that a user has history, but that **order contains predictive information beyond stable user–video affinity, popularity, and impression context**. Here the sequence encoder is an additional causal feature generator for an impression ranker, not a replacement for candidate and context features.

## When to use / avoid
**Use when** timestamp order is reliable; users commonly have meaningful pre-impression histories; and evolving topics, session intent, revisitation, or transitions among videos plausibly matter. Prefer it when a non-sequential embedding/MLP ranker leaves errors concentrated among users with moderate or long histories.

**Avoid or defer when** histories are extremely sparse, event order is ambiguous, or labels are dominated by durable user–video preference. A transformer can then add variance and serving cost without useful incremental signal. SASRec’s original analysis distinguishes sparse settings, where parsimonious methods can be advantageous, from denser settings that can better support richer sequence models. ([cseweb.ucsd.edu](https://cseweb.ucsd.edu/~jmcauley/pdfs/icdm18.pdf?utm_source=openai))

Do not use post-impression watches, clicks, dwell updates, future profile fields, or an event with an unresolved tie relative to the scored impression. If ordering cannot be made deterministic and causal, drop or conservatively truncate the ambiguous event.

## Requirements and implementation
1. Construct one training row per displayed impression with an immutable `impression_time` and split boundary.
2. For that row, select only eligible user events with `event_time < impression_time`; enforce a deterministic secondary ordering only for events that are safely before the impression.
3. Keep the last `L` eligible video IDs, left-pad to length `L`, and retain a padding mask. Start with one event type (for example, completed or sufficiently engaged video views) before mixing heterogeneous behaviors.
4. Map rare/unseen video IDs to an OOV token. Use the same vocabulary policy at train, validation, and test time; never fit encodings using held-out labels.
5. Form token inputs as video embedding + learned position embedding. If elapsed time is trustworthy, optionally add bucketed recency/age embeddings; this is an empirical extension rather than a result established by the cited SASRec paper.
6. Apply `N` Transformer blocks: pre/post-normalization consistent with the implementation, multi-head self-attention, causal mask, padding mask, residual path, dropout, and pointwise feed-forward sublayer. Confirm by unit test that changing an event after a cutoff cannot alter the representation at that cutoff.
7. Extract the hidden state of the final non-padding token. For an empty history, use a learned empty-history representation rather than accidentally reading a padded token.
8. Concatenate sequence state with user ID, candidate video ID, and impression-time metadata/context embeddings. Score with a 2–3 layer MLP. Train with the same pointwise or listwise objective, negative-sampling policy, and temporal split as the baseline.

The original SASRec implementation describes timestamp-sorted interaction sequences and a stack of embedding, self-attention, and pointwise feed-forward components. ([github.com](https://github.com/kang205/SASRec?utm_source=openai))

## Starting configuration and expected effects
Use a deliberately small initial model:

- history length `L`: 50; test 20, 50, 100, and 200 only if the data supports it;
- video/sequence width: 64 or 128;
- Transformer blocks: 1–2;
- attention heads: 2–4, with width divisible by head count;
- feed-forward width: 2–4× model width;
- attention/residual dropout: 0.1–0.3; start at 0.2;
- AdamW-style optimization with validation-based early stopping; tune learning rate jointly with batch size rather than assuming a portable value.

Expected effect: if recent ordered behavior is informative, GAUC may improve through better within-user discrimination, and nDCG@5 may improve when the representation moves the currently relevant candidate into the top ranks. Do **not** presume either metric will improve, or infer a magnitude from SASRec benchmarks: the target here is impression ranking with side information and potentially different exposure bias. Empirically, gains should be largest in cohorts with sufficient valid history and should diminish or reverse for empty/short-history users.

## Diagnostics and risks
- **Future leakage:** suspiciously large offline lifts, especially near split boundaries; train-only replay should show every history event precedes its impression. Audit sampled rows with raw timestamps, IDs, and split membership.
- **Target/impression contamination:** a scored video appearing as a “past” token at the same timestamp, or event logging delayed relative to serving, can leak the label pathway. Require a documented event-time semantics and safety lag if needed.
- **Sequence truncation mismatch:** long-history users improve while medium-history users worsen, or metrics shift sharply with `L`. Compare cohorts by eligible-history length and measure the fraction truncated.
- **Attention overfitting:** training loss improves but validation GAUC/nDCG@5 falls; reduce width/layers, increase dropout, shorten `L`, or return to the non-sequential baseline.
- **Position/recency dependence:** improvement disappears when timestamps are shuffled within user but IDs remain unchanged. This may be a valid recency effect, but verify it survives temporal holdout and does not proxy logging artifacts.
- **Serving cost:** self-attention over length `L` has quadratic attention work in `L`; keep `L`, depth, and width small, batch candidate scoring by shared user history, and cache the pre-impression sequence state when multiple candidates share the same request.
- **Candidate mismatch:** a generic final-state history may miss candidate-specific relevance. If sequence signal exists but top-rank lift is weak, compare later with a candidate-conditioned history-attention architecture under the same causal inputs.

## Cheapest check and clean experiment
**Cheap train-only check:** create two representations for each sampled training impression: the actual causal prefix and a prefix with video IDs randomly permuted within that user while retaining length and timestamps. Train the same shallow scorer for a short, fixed budget. If the true-order version does not beat the permuted-order version on a later-in-time validation slice, sequence order is not yet justified; retain simpler history aggregates or investigate timestamp quality.

**Clean single-variable experiment:** hold split, eligible-event definition, history length, candidate/context features, loss, negatives, optimizer budget, and seeds fixed. Compare (A) the embedding-MLP ranker using existing causal aggregate/history features against (B) the identical model plus a 1-layer, width-64, 2-head causal sequence encoder. Report GAUC and nDCG@5 overall and by history buckets: 0, 1–4, 5–19, 20–49, and 50+ prior events. Also report p50/p95 training and inference latency plus the rate of truncated histories. This isolates the encoder rather than conflating it with larger embeddings or feature changes.

## Related cards and sources
Related cards: `dataset.interaction_log_schema`, `dataset.inventory_and_splits`, `evaluation.within_user_metrics`, `task.experiment_protocol`, `task.leakage_policy`, `features.entity_id_embeddings`, `features.causal_behavior_history_features`, `architecture.embedding_mlp_ranker`, `architecture.candidate_conditioned_history_attention`.

Primary source: Kang, W.-C., and McAuley, J. (2018), *Self-Attentive Sequential Recommendation*, ICDM 2018, pp. 197–206, DOI `10.1109/ICDM.2018.00035`. ([cseweb.ucsd.edu](https://cseweb.ucsd.edu/~jmcauley/pdfs/icdm18.pdf?utm_source=openai))

### Audited web sources

- Self-Attentive Sequential Recommendation: <https://cseweb.ucsd.edu/~jmcauley/pdfs/icdm18.pdf?utm_source=openai>
- GitHub - kang205/SASRec: SASRec: Self-Attentive Sequential Recommendation · GitHub: <https://github.com/kang205/SASRec?utm_source=openai>
