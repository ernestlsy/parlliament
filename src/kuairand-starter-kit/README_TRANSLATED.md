# KuaiRand-Pure Starter Kit

## Dependencies

Python 3.9+ and numpy. **Nothing else.** No torch, pandas, or sklearn required.

## Data

Download from [https://kuairand.com](https://kuairand.com) (Direct Zenodo link, no registration required):

```bash
# Run inside the Starter Kit directory. Unpacking yields ./KuaiRand-Pure/
wget https://zenodo.org/records/10439422/files/KuaiRand-Pure.tar.gz
tar xzf KuaiRand-Pure.tar.gz
```

## Running

```bash
python3 baseline.py --model fm
```

`--data_dir` defaults to `./KuaiRand-Pure/data`; specify explicitly if your data is stored elsewhere.

`--model` options include `fm` (official baseline) / `pop` (trivial baseline) / `random` (lower bound, used for evaluation code sanity checks).
FM takes about 40 seconds in total (CPU, single core).

## Task Definition (Evaluation logic is locked; do not modify)

|  |  |
| --- | --- |
| Task | **In-user Ranking** — Rank only the impressions each user receives in the evaluation set; no global candidate retrieval. |
| Relevance Label | `long_view` (native column, 0/1) |
| Metrics | `GAUC`, `nDCG@5`; **Primary Score = Average of both** |
| Data Split | train `20220408–20220421` / valid `20220422–20220428` / test `20220429–20220508` |
| Zero-positive Users | nDCG is recorded as 0.0 and included in the average; GAUC only evaluates users with `0 < positive count < impression count`, weighted by positive count. |
| nDCG gain | `2^rel − 1` (equivalent to identity under binary labels) |

Implementation can be found in `evaluate.py`; all conventions are documented in the file header comments.

## Baseline Ladder

Scores on the test set. **The goal is to beat the FM row.**

|  | GAUC | nDCG@5 | primary |
| --- | --- | --- | --- |
| random (lower bound, sanity check) | 0.4996 | 0.4511 | 0.4753 |
| item popularity (trivial) | 0.6308 | 0.5121 | 0.5715 |
| **FM (official baseline)** | **0.6610** | **0.5282** | **0.5946** |

### ⚠️ True Range of Metrics: The ceiling for nDCG@5 is 0.729, not 1.0

Across the 23,875 users in the test set:

|  | Proportion | Impact on Metrics |
| --- | --- | --- |
| All-negative Users (none of the impressions are `long_view`) | **27.1%** | nDCG is strictly **0**, no model can fix this; excluded from GAUC. |
| All-positive Users | **9.2%** | nDCG is strictly **1**; excluded from GAUC. |
| Users with Discriminative Labels | **63.7%** | The effective sample pool for GAUC. |

Therefore, using true labels as prediction scores (oracle, perfect ranking) yields at best:

|  | random | FM baseline | **oracle upper bound** | Captured headroom by FM |
| --- | --- | --- | --- | --- |
| GAUC | 0.4996 | 0.6610 | **1.0000** | 32.3% |
| nDCG@5 | 0.4511 | 0.5282 | **0.7289** | 27.8% |
| **primary** | 0.4753 | **0.5946** | **0.8645** | **30.7%** |

**Please evaluate progress using the oracle as the denominator.** Seeing 0.5946 and assuming "it is still far from the full score of 1.0" is a misjudgment—
the baseline has already captured ~30% of the usable headroom, leaving a remaining headroom of 0.27 rather than 0.41.

The std of FM across 5 random seeds is **0.0008**. Based on this, the convergence criterion is set to **ε = 0.002 (≈2.5σ), N = 3**:
convergence is declared if the validation primary score improves by no more than 0.002 for 3 consecutive iterations.

> Sanity Check: If running `--model random` on your evaluation harness does not yield primary ≈ 0.475 (±0.001), your harness is faulty—fix it first.

## Submission Format

CSV format with headers, where each row corresponds to one row in the evaluation set:

```
row_id,user_id,video_id,score
0,0,7531,-3.34176
1,0,4214,-1.4955
...

```

| Field | Description |
| --- | --- |
| `row_id` | Continuously increasing integer starting from 0, corresponding to the row order of `data.load()[split]` (deterministic: reads `log_standard_4_08_to_4_21_pure.csv` first, then `log_standard_4_22_to_5_08_pure.csv`, preserving original file order after date filtering). |
| `user_id` / `video_id` | Redundant fields, used exclusively for alignment verification. |
| `score` | The predicted score assigned by your model to that row (any real number; only relative values matter). Must not contain NaN / Inf. |

> **Why `row_id` is mandatory:** The pair `(user_id, video_id)` is **not unique** in the evaluation set—
> 3.06% of pairs in the test set are duplicates, repeating up to 12 times. Hence, it cannot serve as a primary key.

Generation and validation:

```bash
python3 submit.py --make  --split test  submission.csv    # Generate a sample submission using the official FM baseline
python3 submit.py --check --split test  submission.csv    # Validate format and alignment
python3 submit.py --score --split valid submission.csv    # Validate and score (usable for local validation)

```

`--check` will reject: incorrect headers, row count mismatch, non-sequential `row_id`, misaligned `user_id`/`video_id` with the evaluation set,
or non-numeric / NaN / Inf values in `score`. **Please run `--check` yourself before submitting.**

## Where to Start Modifying

The list below is **experimentally verified**, not guessed. Dead ends tested by the committee are clearly marked to prevent redundant effort.

### Experimentally Verified: These two directions yield no gains; do not waste iterations here

| Attempted Direction | Result |
| --- | --- |
| **Adding static features** — Connecting all 13 feature domains from CWM (+`music_id`/`video_type`/`upload_type` + 6 user-side coarse buckets) | Primary score **0.5940** vs **0.5950** with 5 domains; indistinguishable within noise, slightly worse. |
| **Increasing model capacity** — Embedding dimension k = 8 / 16 / 32 | 0.5895 / 0.5902 / 0.5887, almost no movement. |

Reason: The `user_id × video_id` cross-features already capture most learnable signals. Coarse buckets like `follow_user_num_range` are redundant in the presence of `user_id`; furthermore, 1.14 million rows cannot support larger capacities. **Capacity and features are not the bottleneck.**

⚠️ Note: **First-order terms of pure user-side features contribute exactly 0 to the score.** Because ranking is performed within individual users, any term that remains constant within a user group does not change relative order (experimentally verified: `item_pop × user bias` and pure `item_pop` yield identical scores to every decimal place). User-side features can only take effect through **cross-terms with item-side features**.

### Unexplored: Headroom should be here

Ranked by likelihood of success (**these directions were not tested by the committee and are left for you**):

1. **Change the Loss Function.** The current implementation uses pointwise logloss, but the metrics (GAUC / nDCG) are **ranking metrics**.
Switching to pairwise (BPR) or listwise (applying softmax over a user's impressions) aligns the target function directly with evaluation metrics. This is the most promising direction.
2. **User History Sequences.** Existing features **completely ignore behavioral sequences**. Each user in KuaiRand has hundreds to thousands of interactions in the training set; interest modeling directions like DIN / SIM remain completely unexplored.
3. **Multi-Task Learning.** The logs contain auxiliary signals like `is_click`, `is_like`, `is_follow`, `is_comment`, `is_forward`, and `play_time_ms`, which can serve as auxiliary tasks to support the primary `long_view` task.
4. **Watch Time Modeling.** This is the core contribution of [CWM](https://github.com/hyz20/CWM): treating watch time via **censored regression** (since true watch time is truncated when videos finish, using a one-sided loss instead of squared error). This is a research-worthy direction.
5. **Model Architecture Switch.** DeepFM / DCN / xDeepFM. Since model capacity is proven not to be the bottleneck, **place priority after items 1–4**.
6. **Temporal Features & Distribution Shift.** Modeling `hourmin`, `date`, and addressing distribution shift between train and test.
7. **Unbiased Validation (Advanced).** `log_random_4_22_to_5_08_pure.csv` contains randomly exposed impression logs (1.18 million rows) and can be used as an additional unbiased validation set to verify whether the model overfits to biased traffic.

## Using Your Own Models (Including CWM)

`evaluate.py` is fully decoupled from the model architecture; it only requires three equal-length arrays:

```python
from evaluate import evaluate
print(evaluate(user_ids, labels, scores))   # scores can come from any model

```

* `user_ids`: user_id for each row in the evaluation set
* `labels`: `long_view` label for each row (0/1)
* `scores`: Predicted real scores from your model (relative ranking only)

You can bypass `baseline.py` entirely and use PyTorch, LightGBM, or [CWM](https://github.com/hyz20/CWM)'s xDeepFM—simply pass the final `scores` into `evaluate()`. **The evaluation criteria are uniquely defined by `evaluate.py`.**

> Note when using CWM: It depends on `torch==1.6.0` (a 2020 release that likely fails to install on modern GPUs), its loss optimizes counterfactual watch time, and its evaluation label is a reconstructed `long_view2`. Treat it as an **advanced reference** for paper research rather than a starting point.

## Files

|  |  |
| --- | --- |
| `evaluate.py` | Metric calculations + all evaluation conventions. **Do not modify.** |
| `data.py` | Data loading, official splits, feature encoding. Modify here to add features. |
| `baseline.py` | Implementation of 3 baselines. FM is the target to beat. |
| `baseline_scores.json` | Official benchmark scores + seed variances + convergence parameters. |
| `submit.py` | Submission generation and verification tool. |
| `ablation_features.py` | Feature ablation scripts to reproduce the "adding features yields no gain" findings. |