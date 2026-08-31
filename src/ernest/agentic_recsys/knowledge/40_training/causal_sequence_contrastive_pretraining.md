# Causal sequence-view contrastive pretraining before ranking fine-tuning

## Summary and mechanism
Pretrain the *same causal history encoder* used by the ranker on two independently augmented views of each user’s history prefix, then initialize supervised ranking fine-tuning from that checkpoint. For a scored impression at time \(t\), form the source sequence only from events strictly before \(t\); encode two views with shared weights; project and L2-normalize their pooled representations; and apply an in-batch InfoNCE loss so the two views of one prefix agree while other prefixes act as negatives. Then discard the projection head and fine-tune the encoder jointly with the candidate-ranking head.

The assumption is that mild omissions or masking preserve stable user intent, while the encoder should remain sensitive to meaningful recency and order. Sequence-view contrastive learning with crop, mask, and reorder augmentations was introduced for sequential recommendation by CL4SRec; this card deliberately excludes reordering because a causal video-history ranker treats order as signal. S3-Rec likewise motivates pretraining sequential representations to address sparse supervised signals. ([arxiv.org](https://arxiv.org/abs/2010.14395?utm_source=openai))

## When to use / avoid
**Use** when the production candidate is `architecture.causal_sequence_transformer_ranker`, histories are available as ordered pre-impression prefixes, and supervised training is unstable or loses to a simpler ID-plus-MLP baseline. It is most plausible when many events are unlabeled for the final ranking objective but still describe behavior.

**Avoid** it when augmentation changes intent—for example, a short session whose final rare video is decisive—or when the available batch size/compute cannot support a properly controlled pretrain-versus-no-pretrain comparison. Do not expect contrastive pretraining to be universally superior: published work also reports that optimization-focused regularization can match self-supervised sequential recommenders without pretraining or strong augmentation. ([arxiv.org](https://arxiv.org/abs/2308.10347?utm_source=openai))

## Requirements and implementation
1. Build one immutable training record per impression: `(user_id, impression_time, ordered history IDs/times with event_time < impression_time, candidate video embedding, label)`. Enforce the strict inequality before *any* split, augmentation, cache, or negative construction.
2. Pretrain on train-period prefixes only. Use the causal attention mask in both views; right-pad and pass a padding mask. Never use the scored candidate, its outcome, or events at/after its impression time in either view.
3. Start with last-50 events (or the ranker’s normal maximum length). Independently apply: contiguous suffix-preserving crop retaining 70–100% of tokens; token masking at 5–15%; and optional short contiguous subsequence view retaining 50–80% only for histories of at least 10 events. Preserve timestamps/relative positions for retained tokens. Do **not** reorder.
4. Pool with the final non-padding hidden state or the encoder’s existing causal summary token. Add a small two-layer projection head only during pretraining. Use cosine similarity, temperature 0.07–0.20, and symmetric InfoNCE over the two directions. Treat the other examples in the global batch as negatives; all-gather embeddings across devices if applicable.
5. Pretrain 5–30 epochs or until validation proxy loss stops improving, then fine-tune all encoder layers and the ranking head. Start fine-tuning at 0.25–1.0× the supervised-only learning rate; use a brief warmup and retain the exact ranking loss, candidates, group batching, and early-stopping rule of the baseline.

These defaults are engineering starting points, not research-established optima. CL4SRec provides evidence for crop/mask-style sequence augmentations in contrastive sequential recommendation; DuoRec cautions that data-level augmentations can fail to preserve semantics. ([arxiv.org](https://arxiv.org/abs/2010.14395?utm_source=openai))

## Starting configuration and expected effects
Start conservatively: 50-token maximum history, suffix-preserving crop with minimum retained length `max(5, ceil(0.7L))`, 10% masking, temperature 0.10, projection width equal to encoder width, contrastive batch size at least 256 effective sequences, and 10 pretraining epochs. For histories under five events, use masking only or skip contrastive loss; aggressive crops produce nearly unrelated views.

The intended effect is a less brittle history representation before labels specialize it for ranking. GAUC may improve if pretraining helps distinguish users’ relative candidate preferences; nDCG@5 may improve if the learned history summary better captures recent intent. Neither direction nor magnitude is guaranteed: report both metrics with confidence intervals, sliced by history length, rather than claiming a generic uplift. Sparse-data benefits are consistent with the motivation and experiments reported for S3-Rec, but transfer to an impression-ranking setup remains an empirical question. ([arxiv.org](https://arxiv.org/abs/2008.07873?utm_source=openai))

## Diagnostics and risks
- **Leakage:** A suspiciously large offline gain, especially concentrated near split boundaries, often indicates post-impression events, future popularity/features, or prefix caching keyed without the impression cutoff. Audit random examples by printing the maximum history timestamp alongside the impression timestamp.
- **Semantic destruction:** If contrastive loss falls while GAUC/nDCG@5 decline, crops or masks may erase decisive recent actions. Compare full-history and augmented-view cosine similarity by history length; very low similarity for short histories is a warning.
- **False negatives:** In-batch negatives can include near-identical histories or the same user. Log duplicate-user and high-overlap rates; optionally exclude same-user negatives as an ablation, but keep that policy fixed across runs.
- **Representation collapse or over-uniformity:** Monitor positive cosine, negative cosine, their gap, embedding norms, and effective rank. Near-identical positive and negative similarities, or exploding temperature/logits, indicates an unhealthy objective.
- **Compute regression:** Track wall-clock, accelerator memory, and downstream improvement per extra training hour. Contrastive methods commonly need larger effective batches, while pretraining adds a separate training stage. ([arxiv.org](https://arxiv.org/abs/2308.10347?utm_source=openai))

## Cheapest check and clean experiment
**Cheap train-only check:** take held-out *training-period* prefixes, create two views under the proposed policy, and measure (a) retained-last-event rate, (b) token-overlap distribution, (c) positive-versus-random-pair cosine gap from an untrained and a briefly pretrained encoder, and (d) violations of `max(history_time) < impression_time`. Reject settings with any violation or with frequent empty/near-empty views before spending on ranking runs.

**Clean single-variable experiment:** hold architecture, data split, candidate sets, ranking loss, optimizer schedule, total fine-tuning updates, seeds, and early stopping fixed. Compare: **A)** supervised-only initialization; **B)** identical model after contrastive pretraining. Run at least three seeds; select augmentation hyperparameters using validation only; evaluate once on the untouched test set. Add one diagnostic ablation—mask-only versus crop+mask—only after A/B establishes whether pretraining itself helps.

## Related cards and sources
Related IDs: `architecture.causal_sequence_transformer_ranker`, `features.causal_behavior_history_features`, `task.leakage_policy`, `dataset.inventory_and_splits`, `evaluation.within_user_metrics`, `task.experiment_protocol`, `training.group_complete_stratified_minibatching`, `objective.user_normalized_binary_cross_entropy`, `objective.bce_lambdarank_ndcg5_hybrid`.

Primary sources: Xie et al., *Contrastive Learning for Sequential Recommendation* (ICDE 2022), DOI `10.1109/ICDE53745.2022.00099`; Zhou et al., *S3-Rec* (CIKM 2020), DOI `10.1145/3340531.3411954`; Qiu et al., *DuoRec* (WSDM 2022), DOI `10.1145/3488560.3498433`. ([arxiv.org](https://arxiv.org/abs/2010.14395?utm_source=openai))

### Audited web sources

- Contrastive Learning for Sequential Recommendation: <https://arxiv.org/abs/2010.14395?utm_source=openai>
- Enhancing Transformers without Self-supervised Learning: A Loss Landscape Perspective in Sequential Recommendation: <https://arxiv.org/abs/2308.10347?utm_source=openai>
- S^3-Rec: Self-Supervised Learning for Sequential Recommendation with Mutual Information Maximization: <https://arxiv.org/abs/2008.07873?utm_source=openai>
