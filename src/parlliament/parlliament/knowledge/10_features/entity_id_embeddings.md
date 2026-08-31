# User and video identity embeddings with cold/OOV handling

## Summary and mechanism
Map `user_id` and `video_id` to compact, train-fitted integer vocabularies, then learn one embedding table per field. A minimal scorer is `logit = b + b_u[u] + b_v[v] + <e_u[u], e_v[v]>`; it gives different video scores for the same user and is a strong personalization baseline when repeated entities carry signal. This is the neural form of latent-factor collaborative filtering: it assumes future interaction preferences are partly recoverable from prior user–video co-occurrence, not from the ID strings themselves. Matrix-factorization research established this latent user/item representation pattern; large-scale recommendation systems also use learned embeddings for sparse categorical inputs. ([research.google](https://research.google/pubs/deep-neural-networks-for-youtube-recommendations/?utm_source=openai))

Fit vocabularies **only on the training partition**. Reserve index `0` for `UNK_USER` and `UNK_VIDEO`; map every absent, null, malformed, or post-training ID to its field-specific unknown index at validation, test, and serving. Do not create new rows or resize tables outside a deliberate retraining/versioned vocabulary refresh. For `UNK_USER × UNK_VIDEO`, scores should reduce to global/item effects or other permitted features, rather than imply personalized affinity.

## When to use / avoid
**Use** when validation has substantial warm-user and warm-video coverage, IDs are available at scoring time, and a personalized baseline is needed before histories or metadata. It is particularly useful when ranking candidates within a user request, where a user embedding can interact with video embeddings.

**Avoid as the sole model** when most evaluated users or videos are unseen, IDs are missing at inference, or the split deliberately measures new-content/new-user performance without side-feature fallbacks. Collaborative signals are weak for fresh content; content representations are a documented complement in video recommendation cold-start settings. ([research.google](https://research.google/pubs/content-based-related-video-recommendations/?utm_source=openai))

## Requirements and implementation
1. Construct deterministic train-only ID dictionaries after applying the split cutoff. Store vocabulary version, training cutoff, and `UNK=0` convention with the model artifact.
2. Count train interactions per user and video. Feed lookup embeddings and optional scalar log-frequency features; include global, user, and video bias terms if the task/model supports them.
3. Train with the ranking/classification loss used by the task and L2 weight decay on embeddings and biases. Apply stronger shrinkage to low-count rows. One practical, empirical schedule is `lambda_entity = lambda_base * sqrt(n_ref / max(n_entity,1))`, clipped to roughly `1x–10x`; tune it only from training/validation results.
4. Mask or initialize the two UNK embedding rows consistently. A safe default is trainable UNK rows with stronger regularization; compare against fixed-zero UNK rows. Never let the UNK row absorb an accidental mixture caused by inconsistent ID normalization.
5. Log train/validation/test fractions for: warm user + warm video, OOV user only, OOV video only, and both OOV. Report GAUC and nDCG@5 for these slices as well as overall.

## Starting configuration and expected effects
Start with 32-dimensional user and video embeddings; try `16, 32, 64`, and use 128 only if interaction volume and validation support it. Use Adam/AdamW, a modest base embedding weight decay such as `1e-6–1e-4`, batch size appropriate for the candidate/loss implementation, and early stopping on validation GAUC with nDCG@5 as a guardrail. For very sparse entities, prefer smaller dimensions and/or stronger count-aware regularization before adding capacity.

On warm traffic, identity embeddings commonly improve discrimination because they memorize useful collaborative structure; GAUC may improve when pairwise ordering is better, and nDCG@5 may improve when the top of each user’s candidate list becomes more personalized. No magnitude should be presumed: gains can disappear under heavy pair shift, sparse histories, or OOV-heavy evaluation. nDCG@5 can worsen even when GAUC rises if the model learns broad affinity but misorders the few most relevant videos.

## Diagnostics and risks
- **Leakage:** fitting vocabularies, counts, normalizers, labels, or interaction aggregates using validation/test rows leaks future entity existence or popularity. A suspicious sign is unusually strong performance for ostensibly cold entities.
- **OOV collapse:** poor metrics concentrated in OOV-video or both-OOV slices indicate IDs cannot provide the missing information. Add permitted video/user metadata or a popularity/content fallback rather than enlarging embeddings.
- **Rare-entity overfit:** large train–validation gaps, high embedding norms for one-off IDs, or gains confined to frequent entities suggest insufficient shrinkage. Increase regularization, lower dimension, or minimum-frequency bucket rare IDs into UNK/tail buckets.
- **Popularity shortcut:** strong aggregate metrics but weak per-user GAUC or weak tail-video nDCG@5 can indicate video bias/popularity dominates the interaction term. Compare against a bias-only baseline and frequency-stratified slices.
- **Compute/memory:** tables scale as `(n_users + n_videos) × dimension`; optimizer states can multiply memory. Monitor vocabulary growth, lookup latency, and whether distributed/sharded embeddings are actually needed.

## Cheapest check and clean experiment
**Cheap train-only check:** before fitting a neural model, compute train interaction counts and split coverage. Verify zero overlap violations against the intended temporal/entity policy; tabulate OOV rates and candidate-set coverage by split. Then fit a tiny regularized user–video dot-product model on train only and confirm that scores vary across videos for the same warm user while unknown IDs always map to index 0.

**Clean single-variable experiment:** hold split, negatives/candidate lists, loss, optimizer, parameter budget as closely as possible, and early-stopping protocol fixed. Compare (A) video bias/global bias only versus (B) A plus user/video ID embeddings. Evaluate overall and warm/OOV slices for GAUC and nDCG@5. Next, separately compare uniform L2 against frequency-aware L2; do not change dimensions, metadata, or sampling in that comparison. Retain the change only if the intended warm-slice benefit does not conceal unacceptable OOV or tail regressions.

## Related cards and sources
Related cards: `dataset.interaction_log_schema`, `dataset.inventory_and_splits`, `dataset.population_and_pair_shift`, `dataset.user_metadata`, `dataset.video_metadata_and_statistics`, `evaluation.within_user_metrics`, `task.experiment_protocol`, `task.leakage_policy`, `task.prediction_artifact`.

Primary sources: Covington, Adams, and Sargin, *Deep Neural Networks for YouTube Recommendations* (RecSys 2016); Cheng et al., *Wide & Deep Learning for Recommender Systems* (arXiv:1606.07792, 2016); Lee, Kothari, and Natsev, *Content-based Related Video Recommendations* (NeurIPS Demonstration Track 2016). ([research.google](https://research.google/pubs/deep-neural-networks-for-youtube-recommendations/?utm_source=openai))

### Audited web sources

- Deep Neural Networks for YouTube Recommendations: <https://research.google/pubs/deep-neural-networks-for-youtube-recommendations/?utm_source=openai>
- Content-based Related Video Recommendations: <https://research.google/pubs/content-based-related-video-recommendations/?utm_source=openai>
