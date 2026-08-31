# Quantized embedding tables with a full-precision hot-row cache

## Summary and mechanism
Store every user-ID and video-ID embedding row in a quantized backing table, while retaining a bounded subset of high-access rows in FP32 (or the training master precision). For an integer table, use **per-row affine metadata**: a scale and zero-point/bias for each row; dequantize a row for lookup, perform the model computation in the normal compute precision, then requantize only when writing a non-cached row. A cached row is read and updated in full precision; when a row leaves the cache, write its current value back to the low-precision table.

This follows the mixed-precision-cache design studied by Yang et al.: low-precision backing storage plus a smaller FP32 cache, with row-wise scale/bias quantization and cache-resident high-precision updates. Their results establish that cache size, bit width, replacement policy, and rounding jointly determine the memory–accuracy trade-off; do not treat their reported results as a guarantee for GAUC or nDCG@5 on a different ranking task. ([arxiv.org](https://arxiv.org/abs/2010.11305))

For this card, make the cache **static and deterministic**: compute access counts from the training split only, rank IDs by descending count with ID as a stable tie-breaker, and assign the top-K rows of each table to the FP cache. This prevents validation/test exposure from choosing which entities receive better numerical treatment.

## When to use / avoid
**Use** when user/video embedding and optimizer-state memory constrains batch size or embedding width, access frequencies are strongly concentrated, and target kernels support the intended low-precision lookup path. A frequency-based cache assumes frequently accessed IDs have disproportionate influence on optimization and aggregate ranking quality; that assumption should be measured, not presumed.

**Avoid** when the tables already fit comfortably; the extra quantization, cache, and parity-testing path can be net engineering loss. Also avoid aggressive quantization when rare-video or rare-user quality is a primary product requirement, or when test-platform lookup/dequantization behavior is unverified. Static train-frequency caching is deliberately less adaptive than LRU/LFU designs, but is easier to reproduce and audit. The original cache work compared LFU and LRU and found policy and associativity material to accuracy and hit rate. ([arxiv.org](https://arxiv.org/abs/2010.11305))

## Requirements and implementation
1. Freeze train/validation/test splits before counting. Count user-ID and video-ID occurrences **only in training examples**; do not use labels, future interactions, validation rows, or test rows.
2. Choose a cache budget separately per table. Select top-K IDs by `(−train_count, ID)` and persist the selected-ID list with the model artifact.
3. Start with symmetric signed INT8 per row if supported, or affine UINT8 with FP32 scale plus zero-point/bias. Clamp values to the row range before rounding. Per-row metadata matters: the referenced implementation used one scale/bias pair per row for integer quantization. ([arxiv.org](https://arxiv.org/abs/2010.11305))
4. Keep hot rows and, during training, their optimizer moments in FP32. Account for optimizer states explicitly: quantizing only weights may not resolve the memory bottleneck if Adam-style moments remain full precision.
5. For non-hot rows, dequantize on read; apply updates in a higher-precision temporary; requantize on write. If training below FP16, test stochastic rounding as an option, because the primary study found it preferable to nearest rounding in its low-precision embedding experiments. ([arxiv.org](https://arxiv.org/abs/2010.11305))
6. Log: table bit width, per-row metadata bytes, cache bytes, cache-ID checksum, hot-row access coverage, lookup latency, dequantization time, peak memory, and score-drift statistics.

## Starting configuration and expected effects
Start with **INT8 backing rows + FP32 cache for the top 1–5% of rows per table**, chosen by train access count. Use a larger cache for the table with the more concentrated access distribution; do not force user and video tables to share one percentage. Increase cache budget through 1%, 2%, 5%, and 10%; only then test INT4. The published cache study reported a 3× embedding-memory reduction with INT8 plus a 5% full-precision cache on its DLRM/Criteo setup while maintaining its reported accuracy, but that is architecture- and dataset-specific evidence rather than a GAUC/nDCG@5 expectation. ([arxiv.org](https://arxiv.org/abs/2010.11305))

Likely effect: memory should decline substantially, while GAUC and nDCG@5 may remain near the FP baseline if the hot rows cover most training accesses and quantization error is modest. nDCG@5 can degrade before GAUC when small score perturbations reorder close candidates near the top of a user group. Do not claim success from memory reduction alone; accept only after checking both metrics and score/top-k drift.

## Diagnostics and risks
* **Leakage:** hot-set membership built with all data, future windows, validation/test access counts, or label-conditioned counts invalidates evaluation. Persist split hashes and the count-generation query.
* **Rare-entity degradation:** rising errors concentrated in low-frequency users/videos, lower nDCG@5 in rare-ID strata, or increased score error for rows outside the cache suggests insufficient precision or cache coverage.
* **Boundary churn:** many candidate pairs with small FP margins flip order after quantization. Diagnose score absolute error, score correlation, top-5 overlap, and pairwise order agreement on frozen candidate groups.
* **Cache-budget illusion:** report total memory, including scales, zero-points, cache tags/IDs, cache rows, and optimizer state. The original work explicitly included quantization parameters, cache, tags, and LFU counters in memory accounting. ([arxiv.org](https://arxiv.org/abs/2010.11305))
* **Throughput regression:** theoretical byte savings do not ensure lower latency. Profile batch-size-specific lookup/dequantization kernels on the intended Windows/Linux deployment target.
* **Non-determinism:** stochastic rounding, unstable ties, and dynamic cache replacement complicate reproducibility. For the primary experiment, retain the fixed train-only hot set and fixed seeds.

## Cheapest check and clean experiment
**Cheap train-only check:** From a trained FP checkpoint, construct static hot sets using training counts only. Quantize non-hot user/video rows without retraining, run the same frozen validation candidate groups through FP and mixed-precision artifacts, and report total embedding bytes, mean/max absolute score drift, top-5 overlap, pairwise-order agreement, GAUC, and nDCG@5. Stratify every drift and ranking metric by train-frequency decile for both IDs. This cheaply identifies whether numerical error is concentrated in cold rows or near top-k decision boundaries.

**Clean single-variable experiment:** Hold model, seed set, optimizer, train steps, batches, candidate groups, and static train-only hot IDs fixed. Compare FP32 embeddings against INT8 backing plus FP32 cache at one predetermined budget (start at 5%). Retrain end-to-end, evaluate GAUC and nDCG@5 with confidence intervals or repeated seeds, and compare peak memory and wall-clock throughput. Only after this comparison passes should cache size or INT4 be changed; otherwise multiple moving parts obscure the cause of metric drift.

## Related cards and sources
Related cards: `dataset.inventory_and_splits`, `task.leakage_policy`, `features.entity_id_embeddings`, `training.group_complete_stratified_minibatching`, `evaluation.within_user_metrics`, `evaluation.frozen_candidate_group_integrity_audit`, `evaluation.probability_calibration_and_ranking_error_audit`, `task.experiment_protocol`.

Primary source: Yang, Huang, Park, Tang, and Tulloch, *Mixed-Precision Embedding Using a Cache*, arXiv:2010.11305. It describes low-precision embedding storage, a high-precision row cache, row-wise scale/bias quantization, stochastic rounding, and cache-policy trade-offs. ([arxiv.org](https://arxiv.org/abs/2010.11305))

### Audited web sources

- Mixed-Precision Embedding Using a Cache: <https://arxiv.org/abs/2010.11305>
