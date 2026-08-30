# Benchmark & Scoring — KuaiRand-Pure

Distilled from `PROBLEM.md` (official statement, last updated 27 Aug 2026) and
`src/kuairand-starter-kit/README.md` (Chinese; `README_TRANSLATED.md` is the English copy).
Where the two disagree, the starter kit's `evaluate.py` is authoritative — see
[Known contradiction](#known-contradiction-in-problem-statement).

---

## 1. The task

| | |
|---|---|
| Domain | Short-video feed (Kuaishou), KuaiRand-Pure |
| Task form | **Within-user ranking** — rank each user's logged impressions. NOT full-catalog retrieval. |
| Relevance label | `long_view` (native 0/1 column) |
| Metrics | `GAUC`, `nDCG@5` |
| Primary score | `mean(GAUC, nDCG@5)` |
| Scale | 1.4M interactions, 27K users x 7.6K items |

**Splits** (date-based, fixed, from the two `log_standard_*` files):

| Split | Dates | Rows |
|---|---|---|
| train | 20220408–20220421 | 1,141,112 |
| validation | 20220422–20220428 | 124,909 |
| test (hidden) | 20220429–20220508 | 170,588 |

Develop on train + validation only. Hidden test scored **once**, on the final submission.

**Metric conventions** (pinned in `evaluate.py`, do not change):
- Zero-positive users: nDCG counted as **0.0** and **included** in the average.
- GAUC: only users with `0 < positives < impressions`, weighted by positive count.
- nDCG gain: `2^rel − 1` (identity under binary labels).

---

## 2. Baselines and the real ceiling

Official baseline = organizer-provided FM (k=16, lr=0.001, 5 categorical fields, numpy only, ~40s CPU).
**This is the thing to beat** — not any baseline we build ourselves.

| | GAUC | nDCG@5 | primary |
|---|---|---|---|
| random (harness self-check) | 0.4996 | 0.4511 | 0.4753 |
| item popularity (trivial) | 0.6308 | 0.5121 | 0.5715 |
| **FM baseline — TEST** | **0.6610** | **0.5282** | **0.5946** |
| **FM baseline — VALIDATION** | **0.6674** | **0.5357** | **0.6016** |
| oracle (true labels as scores) | 1.0000 | 0.7289 | 0.8645 |

> **Use the validation row when comparing our runs.** Our agent scores on validation
> (124,909 rows). Baseline validation primary is **0.6016**, not 0.5946.

**The metrics do not span [0,1].** On test, 27.1% of users have zero positives
(nDCG = 0 for any model, ever) and 9.2% are all-positive. Only 63.7% are
discriminative and enter GAUC. So:

- Attainable range is `0.4753 → 0.8645`, width **0.389**.
- FM already captures ~31% of it. Remaining headroom is **0.27**, not 0.41.
- Judge progress against **0.8645**, never against 1.0.

Baseline seed variance: std **0.0008** over 5 seeds. Hence the convergence epsilon
below is ~2.5σ.

---

## 3. Convergence rule (this decides our score)

> A run is converged when validation primary has not improved by more than
> **ε = 0.002** over the last **N = 3** consecutive iterations — or on the
> 50-iteration cap, or the 6h wall-clock ceiling, whichever comes first.

**What gets scored:** the **validation-best checkpoint at the convergence point**,
evaluated once on hidden test. Not the peak reached afterwards, not the final score.
Nothing should run after the stop condition triggers.

**Scoring formula:**
```
delta(m)      = score_agent(m) − score_baseline(m)      # on hidden test
score_dataset = mean over m of delta(m)                  # m in {GAUC, nDCG@5}
```
Equivalently: our test primary minus 0.5946. Falling short is scored **continuously**,
not as a disqualification — a negative delta still scores, just worse.

---

## 4. Constraints

| | |
|---|---|
| Iteration cap | **50 per benchmark run** (hard) |
| Wall-clock | **6h per run** (backstop) |
| Compute | Deliberately not binding. 100 baseline iterations ≈ 28 min, single CPU core, no GPU. |
| Tokens / GPU-hours | Not capped, but **reported and scored** under Feasibility |
| Hidden test | No access during development |
| External data | **Hard rule: none.** Only KuaiRand datasets. No pretraining on any other dataset, no weights trained on these test labels. |
| Everything else | Open — any OSS library, any paper, any public solution, any pretrained weights (subject to the above) |

**Benchmarks:** KuaiRand-Pure is required and is **100% of the primary score**.
KuaiRand-1k (11.7M) and KuaiRand-27k (322M) are optional bonus; skipping them costs nothing.

---

## 5. Submission format

CSV with header, one row per evaluation-split row:

```
row_id,user_id,video_id,score
0,0,7531,-3.34176
1,0,4214,-1.4955
```

- `row_id` — 0-based, strictly increasing, indexes `data.load()[split]` row order.
  Deterministic order: read `log_standard_4_08_to_4_21_pure.csv` first, then
  `log_standard_4_22_to_5_08_pure.csv`, filter by date, preserve in-file order.
- `user_id` / `video_id` — redundant, alignment check only.
- `score` — any real number, only relative order matters. NaN/Inf rejected.

**Why `row_id` is mandatory:** `(user_id, video_id)` is **not unique** — 3.06% of test
rows are repeated pairs, up to 12x. It cannot be a key.

```bash
python3 submit.py --make  --split test  submission.csv
python3 submit.py --check --split test  submission.csv
python3 submit.py --score --split valid submission.csv
```
Always run `--check` before submitting.

---

## 6. Where the headroom is

The organizers ran these and published the results. **Do not re-spend iterations here:**

| Tried | Result |
|---|---|
| More static features — all 13 CWM feature fields (+`music_id`/`video_type`/`upload_type` + 6 coarse user buckets) | primary **0.5940** vs 5-field **0.5950**. Within noise, slightly worse. |
| More capacity — embedding k = 8 / 16 / 32 | 0.5895 / 0.5902 / 0.5887. Flat. |

Reason: the `user_id x video_id` cross already absorbs most learnable signal.
Buckets like `follow_user_num_range` are redundant given `user_id`, and 1.14M rows
won't support more capacity. **The bottleneck is neither features nor capacity.**

**Critical structural fact:** pure user-side first-order terms contribute **exactly zero**.
Ranking happens within a user, so any term constant within a user cannot change the
intra-group order (verified: `item_pop x user_bias` scores identically to bare `item_pop`).
User-side features can only act through **crosses with item-side terms**.

**Unexplored — organizers' own priority order** (they have not tested these):

1. **Ranking loss.** Current objective is pointwise logloss; the metrics are ranking
   metrics. Pairwise (BPR) or listwise (softmax over the user's impressions) aligns
   objective with evaluation. Organizers rate this most likely to work.
2. **User history sequences.** Current features use no behavioural sequence at all.
   Each user has hundreds to thousands of train interactions. DIN / SIM-style interest
   modelling is completely untouched.
3. **Multi-task.** `is_click`, `is_like`, `is_follow`, `is_comment`, `is_forward`,
   `play_time_ms` as auxiliary tasks for the `long_view` main task. ESMM-style.
4. **Watch-time modelling.** CWM's contribution: censored regression on watch time
   (a completed play truncates true watch time, so use a one-sided loss, not squared error).
   Research-depth direction. Ref [4].
5. **Different model.** DeepFM / DCN / xDeepFM. **Deprioritised below 1-4** — capacity
   is measured not to be the bottleneck.
6. **Time features & drift.** `hourmin`, `date`, train→test distribution shift.
7. **Unbiased validation (advanced).** `log_random_4_22_to_5_08_pure.csv` is a
   randomized-exposure log (1.18M rows) — usable as an extra unbiased validation set to
   check whether the model only overfits biased traffic. Also enables off-policy /
   counterfactual evaluation.

CWM caveats if used: depends on `torch==1.6.0` (2020, likely uninstallable on modern GPUs),
optimises counterfactual watch time, evaluates on its own rebuilt `long_view2` label, ships
no Recall implementation. Advanced reference, not a starting point.

---

## 7. Judging criteria

| Criterion | Weight | What it measures |
|---|---|---|
| Technical Execution | 35% | Primary metric (converged test delta) + Robustness |
| Innovation & Problem Insight | 20% | What the agent chose to target and **why**; originality in drawing on published methods |
| Impact & Relevance | 20% | **Autonomy** — measured primarily by count of manual interventions |
| Feasibility & Practicality | 15% | Token consumption + agent wall-clock, in 3 coarse tiers |
| Presentation & Communication | 10% | Final event only |

Notes that change what we build:

- **Robustness is not failure count.** Judged on *how the agent handles* a failure —
  recover, retry, or route around it — so long runs neither crash, stall, nor diverge.
  A capable agent may fail often on genuinely hard problems and still score well.
- **Innovation is judged on reasoning, not implementation.** What it chose to try and
  why, across the *full* stack (features, architecture, training, evaluation loop), and
  how well it drew on real published methods rather than naive baseline tweaks.
- **Autonomy is counted in manual interventions.** Fully autonomous scores highest;
  a well-instrumented semi-automated pipeline needing a handful of interventions is
  explicitly acceptable.
- **Feasibility is gated.** Only scored among submissions whose hidden-test primary
  **exceeds the baseline** — otherwise stopping after 3 iterations would look cheapest.

---

## 8. Deliverables

1. **Written project description** (Devpost) — how it addresses the problem, dev tools,
   APIs, libraries/frameworks, datasets/assets.
2. **Public code repo** — well-structured commented code, README with overview, setup,
   reproduction steps, a reflection on limitations and what we'd improve, and team
   member contributions.
3. **Run & iteration logs** — per iteration: hypothesis (what and why), code diff,
   resulting metrics, error/recovery events and how they were handled. Plus a summary
   of manual-intervention count.
4. **Final submission & results summary** — model output in starter-kit schema; results
   table with validation-best GAUC/nDCG@5 and absolute delta over baseline; reported
   resource usage (total input+output tokens, total agent wall-clock, iterations used
   out of 50, GPU-hours if any).

No video required; ~3 min video recommended, otherwise a detailed report is
"highly encouraged".

> Deliverable 3 maps almost exactly onto the journal we already emit. See `docs/STATUS.md`.

---

## Known contradiction in problem statement

`PROBLEM.md` §2.3 "Limits" says:

> KuaiRand-Pure: NDCG@10 / Recall@50, click = positive (fixed)

**This is stale.** Every other part of the document and the starter kit pin
`long_view` as the label and `GAUC` / `nDCG@5` as the metrics. Appendix A.4 even
explains why Recall was dropped: each user has only ~5 logged impressions in the
evaluation split, so Recall@50 is 0.999+ for every model including random.

**Follow `evaluate.py`.** The statement says so explicitly: "评分口径由 `evaluate.py`
唯一决定" — the scoring convention is determined solely by `evaluate.py`.

---

## Prior art named by the organizers

- **MLE-Bench** (OpenAI, arXiv:2410.07095) — 75 Kaggle competitions, standard eval suite.
- **AIDE** (Weco AI, arXiv:2502.13138) — frames ML engineering as code optimization,
  tree search over solution space. Our `max_debug_attempts=3` mirrors its `max_debug_depth`.
- **AI-Scientist-v2** (Sakana, arXiv:2504.08066) — agentic tree search for hypotheses,
  experiments, write-up.
- **CWM** (Zhao et al., KDD 2024) — https://github.com/hyz20/CWM
