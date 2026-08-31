# ParLLiaMent Agentic Recommender System — Context Handoff

Last updated: 2026-09-01 (Asia/Singapore)

This document is intended to transfer the working context for `src/parlliament` into another Codex
conversation or a fork of this repository. It describes the current implementation, the decisions
made during development, the run protocol, important invariants, known limitations, and the most
useful continuation points. Treat the code as authoritative if this document and the fork diverge.

## 1. Goal and scope

ParLLiaMent is an autonomous, sequential, hypothesis-driven system for developing a recommender model on
KuaiRand-Pure. It uses LLM-backed research and coding roles, but the host process owns experiment
counting, paths, contracts, evaluation, convergence, and submission generation.

The main constraints are:

- Official experiments are scarce and run strictly one at a time.
- Only successfully scored validation experiments consume experiment IDs.
- Planning, train-only feature screening, retries, and abandoned attempts do not count as official
  experiments.
- Generated code may change only the experiment-owned `data.py`, `model.py`, `train.py`, and
  `config.json` files, according to role ownership.
- The evaluator, canonical row order, ground truth, experiment journal, and stopping rule remain
  system-owned.
- The seed is parent experiment 0 and is not treated as measured evidence.
- Work has intentionally remained inside `src/parlliament`; `src/kuairand-starter-kit` is a read-only
  reference for metric and submission conventions.

## 2. Benchmark definition

Task: rank already logged KuaiRand-Pure impressions for `long_view`. This is within-user ranking over
the supplied impression rows, not full-catalog retrieval.

Official temporal split:

| Split | Dates | Purpose |
|---|---|---|
| Train | 2022-04-08 through 2022-04-21 | Model fitting and internal train-only screening |
| Validation | 2022-04-22 through 2022-04-28 | Official experiment scoring and checkpoint selection |
| Test | 2022-04-29 through 2022-05-08 | Final submission predictions only |

Official metrics:

- `GAUC`: positive-count-weighted mean of per-user ROC AUC for users with both label classes.
- `nDCG@5`: macro mean per-user nDCG at five.
- `primary = (GAUC + nDCG@5) / 2`.

Only `primary` drives experiment selection and convergence. Diagnostic metrics are available to the
Evolution Judge but do not change the official score.

Official validation FM baseline used in submission reports:

| Metric | Baseline |
|---|---:|
| GAUC | 0.6674 |
| nDCG@5 | 0.5357 |
| primary | 0.6016 |

## 3. Current convergence and submission policy

The convergence rule was deliberately revised several times. The current and authoritative rule is
implemented in `parlliament/journal.py`:

1. Consider every successfully scored experiment `n` with primary score `x` as an anchor.
2. Its target is `x + 0.002`.
3. Inspect the next three successfully scored experiments: `n+1`, `n+2`, and `n+3`.
4. Stop after `n+3` if all three scores are strictly less than `x + 0.002`.
5. If any of the three reaches or exceeds `x + 0.002`, that anchor does not cause convergence.
6. Abandoned attempts are not present in this score sequence.
7. Independently stop at the configured experiment cap, default 50.

In pseudocode:

```python
for every anchor n with three later scored experiments:
    threshold = score[n] + 0.002
    if all(score[j] < threshold for j in (n + 1, n + 2, n + 3)):
        stop
```

The boundary is strict: a later score exactly equal to `x + 0.002` prevents convergence for that
anchor.

The submitted model is always the experiment with the highest validation `primary` score observed
before stopping. It need not be the last experiment or the experiment that triggered convergence.
The run result exposes both the final score and the submission choice through
`converged_score`/`last_experiment_id` and `submitted_score`/`submitted_experiment_id`.

## 4. Terminology

These terms are intentionally distinct:

- **Generation**: one planning/tournament cycle. Odd generations use Draft mode; even generations
  use Improve mode.
- **Hypothesis**: the proposed change, its parent experiment, expected metric effects, evidence,
  exact ablation, confidence, and risk metadata.
- **Attempt**: one implementation/execution effort for a hypothesis. It has an opaque attempt ID.
  It may undergo several code-response repairs and up to the configured debug attempts.
- **Abandoned attempt**: an attempt that never produced valid official predictions. It is journaled
  for audit and replacement planning but receives no experiment ID.
- **Experiment**: a successful attempt whose validation artifact passed the fixed evaluator. Only
  these receive sequential IDs and count toward convergence and the cap.
- **Parent 0 / seed**: an unscored code scaffold. It is a legal lineage parent but not measured
  experimental evidence.

## 5. End-to-end workflow

The main control loop is `Overseer.run()` in `parlliament/overseer.py`.

1. Initialize or resume `runs/<run_name>`, validate the selected seed, save `system_config.json`,
   and read the durable journal.
2. If the journal is not already stopped, run or reuse the train-only feature screen.
3. Build current archive context. Draft mode sees the seed and full scored archive. Improve mode uses
   the globally best scored parent as its primary reference.
4. The Feature Analyst summarizes train-only evidence and recent metric weaknesses.
5. The Evolution Judge may request literature. The Librarian retrieves and fetches relevant cards.
6. The informed Judge generates a candidate portfolio (default 12 candidates).
7. The Consultant ranks all candidates head-to-head.
8. The Judge selects exactly one winner. Planning and retrieval do not consume an experiment ID.
9. The Orchestrator converts the winner into an explicit interface contract and role-specific
   instructions. It activates only the necessary code agents.
10. Active code agents return complete final contents of all files they own. The system applies
    allowlisted complete-file replacements; it does not ask agents for context-sensitive patches.
11. The Experimentor runs a small contract probe, then full training in a subprocess under one
    wall-clock budget.
12. Failures are classified and routed only to responsible roles. Repairs are bounded. Persistent
    failures become abandoned attempts and may trigger replacement hypotheses.
13. A successful validation artifact is evaluated by fixed host code. Post-score segment diagnostics
    are attached for internal Judge context.
14. The sandbox is atomically finalized as `experiment_<id>`, its unified diff is journaled, and the
    stopping rule is reevaluated.
15. On stop, cumulative timing is persisted and a submission bundle is generated from the globally
    best validation experiment.

## 6. Agent roles

The LLM-backed roles live in `parlliament/agents.py`:

| Role | Responsibility |
|---|---|
| Feature Analyst | Interprets train-only screen results and diagnostic weaknesses; repairs unsupported evidence references |
| Evolution Judge | Requests research, generates a diverse evidence-citing candidate portfolio, proposes replacements after failures, and selects one winner |
| Consultant | Reviews hypotheses, ranks candidates head-to-head, and resolves/flags proposal issues |
| Orchestrator | Chooses active code roles, fixes the interface contract, and gives each role explicit implementation instructions |
| Feature Engineer | Owns only `data.py`; always receives the deterministic dataset feature schema |
| Model Designer | Owns only `model.py` |
| Trainer | Owns only `train.py` and `config.json` |
| Experimentor | Non-LLM host component that probes, runs, classifies failures, validates artifacts, and invokes fixed evaluation |
| Librarian | Retrieves cataloged Markdown literature and returns/fetches validated document IDs |

An important earlier failure mode was having the code agents interpret a broad hypothesis on their
own. The current Orchestrator must instead provide each active role an `objective`, explicit
`required_changes`, behavior to `preserve`, and `coordination_notes`.

Another earlier failure mode was requesting unified patches from an LLM and receiving “patch context
does not match reference file.” Code agents now return complete file contents. Responses may be
either `{ "files": { ... } }` or the exact flat allowlisted filename mapping. Empty strings,
non-string content, missing owned files, invalid Python/JSON, path escapes, and no-op replacements
are rejected before mutation.

## 7. Seed choices

The CLI flag `--seed-model` selects one of two ParLLiaMent-owned parent-0 scaffolds:

### `simple` (default)

Location: `parlliament/seed/`

- Fresh start, intentionally not the KuaiRand baseline.
- Fields: `user_id`, `video_id`.
- Additive pointwise model with first-order categorical weights and a bias.
- BCE-like training, Adam-style state, default learning rate `0.01`, L2 `1e-6`.
- Batch size 8192, maximum 10 epochs, patience 3.

### `kuairand-baseline`

Location: `parlliament/seed_kuairand_baseline/`

- ParLLiaMent-owned adaptation of the Starter Kit FM scaffold.
- Fields: `user_id`, `video_id`, `author_id`, `tab`, and a training-fitted duration bucket.
- Factorization Machine with 16-dimensional interactions.
- Default learning rate `0.001`, L2 `1e-6`, batch size 8192, maximum 40 epochs, patience 4.

Both seeds are unscored parent 0. Selecting the baseline code does not inject its published score as
an experiment. A run cannot be resumed with a different seed choice.

## 8. Train-only feature screener

`parlliament/research.py` implements feature profiling and screening before official
experimentation.

- It uses only the official training period.
- It creates a temporal holdout by training on earlier training dates and evaluating on the latest
  fraction of training dates (default holdout fraction 0.25).
- “Temporal” means split by event date, not random rows, so the proxy tests forward generalization.
- It profiles raw schemas, missingness/coverage, leakage status, and supported feature groups.
- It caches a dataset fingerprint, feature catalog, screening report, split boundary, and dependency
  versions under `runs/<run>/research/`.
- It never reads official validation rows for model selection, writes official predictions, creates
  a Journal experiment record, or consumes an experiment ID.
- The Feature Engineer receives the raw source schema and canonical derivation recipes so it is less
  likely to invent unavailable fields or aliases.

The screen is a planning aid, not a substitute for official validation. Use `--force-rescreen` when
the dataset or screening implementation changes.

## 9. Evaluation and diagnostics

`parlliament/evaluation.py` is fixed host-side scoring code matching Starter Kit conventions.
Generated experiment code does not own or copy the evaluator.

Every successful experiment retains:

- Official: `GAUC`, `nDCG@5`, `primary`.
- Classification diagnostics: accuracy, balanced accuracy, precision, recall, specificity, F1,
  Matthews correlation, predicted-positive rate, confusion matrix, and selected threshold.
- Ranking diagnostics: global AUC, average precision, Precision@5, Recall@5, MAP@5, MRR@5, and
  HitRate@5.
- Data diagnostics: label prevalence and score distribution summary.
- Segment diagnostics: warm/cold and frequency/activity/context/duration/positive-count slices.

Classification threshold selection maximizes validation F1, with ties broken by balanced accuracy,
accuracy, then the higher threshold. Scores are arbitrary ranking values, so a hard-coded probability
threshold of 0.5 is not assumed.

`parlliament/diagnostics.py` computes segment diagnostics only after official scoring. They remain
in the internal journal and are available to the Evolution Judge, but they are intentionally removed
from submission-facing `iteration_log.json` because they are not required by the submission format.

## 10. Prediction artifacts and test safety

Every successful experiment must produce two NumPy `.npz` files containing exactly the named arrays
`row_ids` and `scores`:

- `predictions_valid.npz`: canonical validation row order; evaluated locally.
- `predictions_test.npz`: canonical test row order; used only to create the final CSV.

Both are emitted in the same training pass after loading the validation-selected checkpoint.
Checkpoint selection and early stopping must use only validation data. The Experimentor enforces
`config["split"] == "valid"`, validates the test artifact's schema/alignment/finiteness, and does not
calculate or expose test metrics.

The final CSV schema is exactly:

```text
row_id,user_id,video_id,score
```

The pair `(user_id, video_id)` is not unique, so `row_id` and canonical row order are mandatory.

## 11. Knowledge base and Librarian

The live literature base is `parlliament/knowledge/`:

- One categorized Markdown file per knowledge card.
- `catalog.jsonl` stores retrieval metadata.
- `manifest.json` stores schema information and canonical document hashes.
- Categories cover task, dataset, features, architectures, objectives, training, evaluation,
  bias/robustness, efficiency, experiment strategy, and papers.
- Task and dataset cards are system-owned because ParLLiaMent has direct access to the task definition and
  supplied dataset.
- Fields previously proposed for `compute_cost`, `leakage_risk`, and `evidence_level` were removed
  from the card schema and Librarian checks because they were not needed for retrieval.

The Librarian uses hybrid retrieval:

1. Deterministic TF-IDF/MMR retrieval produces 10 candidates by default.
2. LLM-assisted query expansion/retrieval contributes another 10 candidates by default.
3. An LLM ranks/selects the final reading list; the system only needs the final document IDs rather
   than separate novelty/strength/risk scores.
4. At most 8 documents and 40,000 fetched characters are added per generation by default.
5. Every fetched ID, path, hash, category, candidate-pool membership, uniqueness constraint, and
   character limit is validated by host code.
6. Retrieved Markdown is untrusted literature context and cannot override fixed task, leakage,
   evaluation, or path rules.

The original agreed design is also recorded in `LIBRARIAN_RETRIEVAL_PLAN.md`.

## 12. Knowledge builder

`parlliament/knowledge_builder.py` and `scripts/populate_knowledge_base.py` implement a manual,
one-time curator. This builder is not run during recommender experiments.

Notable behavior:

- Supports `extend` and confirmation-gated `replace` modes.
- Task and dataset categories are preserved as system-owned and are not generated by the curator.
- The LLM chooses a category card count within configured minimum/maximum bounds rather than being
  forced to return an exact count (defaults 4–10).
- OpenAI Responses hosted web search is enabled by default for HTTP-model generation and citations
  are captured/validated.
- Provider JSON mode is deliberately omitted when hosted web search is enabled because the OpenAI
  API rejects the combination; returned text is still parsed and schema-validated locally.
- Overlong Markdown cards are truncated to the configured character limit instead of failing.
- A card that exhausts its response/repair attempts is dropped and reported instead of crashing the
  full generation job.
- Generation occurs in staging, validates the catalog and hashes, then installs atomically with a
  backup/report/audit log.

Example:

```bash
parlliament-populate-knowledge \
  --model <capable-openai-model> \
  --mode replace --yes \
  --minimum-cards-per-category 4 \
  --maximum-cards-per-category 10
```

## 13. LLM clients and auditing

`parlliament/llm.py` contains:

- `OpenAICompatibleClient`: OpenAI Responses API or compatible Chat Completions API.
- `CommandLLMClient`: local/custom JSON-over-stdio adapter.
- `ScriptedLLMClient`: deterministic testing adapter.
- `AuditedLLMClient`: writes full request/response/error events to run-level and attempt-level JSONL.

Important fixes already present:

- OpenAI JSON mode requests explicitly contain the word “JSON,” avoiding the Responses API 400 that
  requires JSON to appear in input when `text.format.type=json_object` is used.
- HTTP 400 errors retain provider response bodies and request IDs for diagnosis.
- Read timeouts, selected HTTP errors, and network failures use bounded retries and exponential
  backoff.
- `temperature` is omitted for compatibility with models/endpoints that reject it.
- Hosted web search is only allowed with Responses mode and not provider-enforced JSON mode.
- Provider-reported input, output, and total token usage is attached to successful audit events.

All LLM roles share one audited client. `runs/<run>/llm_events.jsonl` includes planning calls that
occur before an attempt directory exists. Attempt-local LLM events are copied to
`experiment_<id>/llm_events.jsonl` or the abandoned-attempt directory.

## 14. Failure handling and durable logging

Failures are classified as semantic logic, contract fulfillment, contract usage, resource/transient,
or timeout failures. The Experimentor records contract and training stdout/stderr. The Overseer
routes a structured failure only to responsible agents.

Defaults:

- Up to 3 debug attempts per experiment.
- One 900-second wall-clock ceiling per experiment attempt chain.
- A repeated resource failure is reclassified as semantic rather than retried forever.
- Up to 2 replacement/backfill hypotheses per abandoned generation slot.
- Abandoned attempts never consume an experiment ID, but all are journaled.

Useful attempt artifacts:

- `attempt_summary.json`: status, stage, failure chain, elapsed time, and log inventory.
- `failure.log`: readable abandonment reason and traceback.
- `patch_history.json`: complete-file response history, including invalid/repaired responses.
- `contract_attempt_<n>.log`: contract probe output.
- `attempt_<n>.log`: full training subprocess output.
- `hypothesis.json`, `orchestrator_plan.json`, and `interface_contract.json`.

Earlier runs were hard to diagnose because abandoned records lacked detailed reasons. Current code
preserves failure stage, exact failure reason, structured reports, tracebacks, and relevant logs.

## 15. Journal, lineage, and evidence rules

`parlliament/journal.py` owns append-only `journal.jsonl` state and calls `flush` plus `fsync` for
durability.

- Scored experiments are sorted by experiment ID.
- The next experiment ID is based only on scored records.
- Abandoned records can have no experiment ID and cannot create gaps or consume IDs.
- Parent references are revalidated against currently available scored experiments before initial
  or replacement hypothesis calls.
- Evidence references use stable available IDs. Invalid or stale IDs are returned to the LLM for
  repair instead of crashing immediately.
- Knowledge-card IDs supplied as evidence are reclassified as literature when appropriate.
- Parent 0 is legal but must never be cited as having measured metrics.

## 16. Submission bundle

`parlliament/submission.py` builds `runs/<run>/submission/` automatically when a run stops. It can
also be refreshed manually:

```bash
parlliament submit runs/run_12
```

Bundle contents:

- `iteration_log.md`: readable per-experiment and abandoned-attempt log.
- `iteration_log.json`: structured hypotheses, final code/config diffs, official/diagnostic metrics
  except segment diagnostics, failures, and recovery actions.
- `results.md` and `results.json`: validation-best GAUC/nDCG@5/primary, absolute deltas over the
  official validation FM baseline, and resource usage.
- `kuairand_pure_submission.csv`: 170,588 canonical test rows from the validation-best experiment.
- `manifest.json`: completeness, selected experiment, row count, output index, and errors.

Submission requirements addressed:

- Per iteration: hypothesis and rationale, applied diff, GAUC/nDCG@5, errors, and recoveries.
- Manual interventions default to `0` / `None`.
- Final output uses `row_id,user_id,video_id,score`.
- Results report absolute metric deltas from the official baseline.
- Resources report provider input/output/total tokens, LLM calls, accumulated run wall-clock,
  counted iterations, and total attempts.

Historical runs created before test artifacts or timing/token capture may have an `incomplete`
manifest. Their missing historical values are not fabricated. New runs can be complete.

## 17. Current run inventory

The following values were read from each run's submission report at handoff time:

| Run | Best experiment | Validation GAUC | Validation nDCG@5 | Validation primary | Bundle |
|---|---:|---:|---:|---:|---|
| run_1 | 1 | 0.6296990819601160 | 0.5192306190663164 | 0.5744648505132162 | incomplete legacy |
| run_2 | 1 | 0.6498691933078030 | 0.5272150703731953 | 0.5885421318404991 | incomplete legacy |
| run_3 | 2 | 0.6689116170510226 | 0.5369580777053825 | 0.6029348473782026 | incomplete legacy |
| run_4 | 1 | 0.6691805131784498 | 0.5364363348416990 | 0.6028084240100744 | incomplete legacy |
| run_5 | 1 | 0.6693696647860992 | 0.5366659554897968 | 0.6030178101379480 | incomplete legacy |
| run_6 | 1 | 0.6693696647860992 | 0.5366659554897968 | 0.6030178101379480 | incomplete legacy |
| run_7 | 1 | 0.6693696647860992 | 0.5366659554897968 | 0.6030178101379480 | incomplete legacy |
| run_8 | 1 | 0.6689116170510226 | 0.5369580777053825 | 0.6029348473782026 | incomplete legacy |
| run_9 | 1 | 0.6670174915586686 | 0.5358419331326125 | 0.6014297123456406 | incomplete legacy |
| run_10 | 3 | 0.6704966836327051 | 0.5376797640751650 | 0.6040882238539351 | incomplete legacy |
| run_11 | 1 | 0.6670174915586686 | 0.5358419331326125 | 0.6014297123456406 | incomplete legacy |
| run_12 | 4 | 0.6707057443888901 | 0.5382868021095627 | 0.6044962732492264 | complete |

Run 12 currently has the strongest reported validation score. Its complete bundle selected
experiment 4 and contains 170,588 test predictions. Its provider-reported LLM usage is 3,627,142
input tokens, 89,403 output tokens, and 3,716,545 total tokens across 87 calls (5 failed calls and
82 successful calls with usage). Its wall-clock field is marked incomplete because that run began
before cumulative timing capture was available. Operationally, Run 12 ended because the agent ran
out of token credits, not because the live loop triggered its convergence rule.

Do not assume old runs' historical stop decisions match the latest convergence rule. `parlliament status`
recomputes convergence from the current Journal implementation, so its retrospective
`"converged": true` result for Run 12 does not describe the historical stop cause.

## 18. Important Python files

| File | Purpose |
|---|---|
| `parlliament/__main__.py` | Enables `python -m parlliament` |
| `parlliament/cli.py` | `run`, `status`, and `submit` command-line interface |
| `parlliament/config.py` | Cross-platform persisted `SystemConfig` and validation |
| `parlliament/schemas.py` | Hypothesis, contract, plan, failure, result, and Journal dataclasses/enums |
| `parlliament/overseer.py` | Sequential lifecycle, tournaments, retries, IDs, stopping, timing, submission |
| `parlliament/agents.py` | Feature Analyst, Judge, Consultant, Orchestrator, and code-agent prompts |
| `parlliament/llm.py` | HTTP/command/scripted clients, retries, JSON parsing, usage auditing |
| `parlliament/research.py` | Dataset schema, profiles, leakage rules, temporal screen, cache manifest |
| `parlliament/librarian.py` | Catalog validation, TF-IDF/MMR, LLM assistance, reading-list fetch/audit |
| `parlliament/knowledge.py` | Always-included knowledge loading |
| `parlliament/knowledge_builder.py` | Manual web-assisted knowledge curation and atomic install |
| `parlliament/sandbox.py` | Safe sandbox paths, copy/finalize, complete-file validation/replacement |
| `parlliament/experimentor.py` | Preflight, subprocess execution, artifact checks, failure classification |
| `parlliament/evaluation.py` | Fixed official metrics plus classification/ranking diagnostics |
| `parlliament/diagnostics.py` | Post-score segment diagnostics |
| `parlliament/journal.py` | Durable records, experiment IDs, convergence, stop reason |
| `parlliament/submission.py` | Per-run logs, result deltas, resource totals, final test CSV |
| `parlliament/seed/*` | Fresh simple parent-0 scaffold |
| `parlliament/seed_kuairand_baseline/*` | Optional five-field FM parent-0 scaffold |

## 19. Installation and common commands

Python requirement: 3.9 or later.

Dependencies:

```text
numpy>=1.23
pandas>=2.0,<3.0
scikit-learn>=1.3,<2.0
```

Install from `src/parlliament`:

```bash
python -m pip install -e .
```

Example OpenAI run:

```bash
parlliament run \
  --workspace /path/to/parlliament-workspace \
  --data-dir /path/to/KuaiRand-Pure/data \
  --run-name run_13 \
  --seed-model kuairand-baseline \
  --model <model-name>
```

PowerShell uses the same arguments; line continuation is backtick rather than backslash. Paths are
handled with `pathlib`, subprocess commands are argument lists rather than shell strings, and the
implementation is intended to work on both Linux and Windows. Avoid copying absolute journal paths
between operating systems; submission lookup reconstructs canonical experiment paths from the run
directory and experiment ID for portability.

Other commands:

```bash
parlliament status /path/to/runs/run_13
parlliament submit /path/to/runs/run_13
python -m unittest discover -s tests -v
```

Never place API keys in this handoff file. The local `.env` may contain private configuration and
must not be copied into prompts or committed without review. The HTTP client accepts `--api-key` or
`OPENAI_API_KEY`.

## 20. Tests and current verification caveat

The test suite covers:

- Metric parity and edge cases.
- Classification/ranking diagnostics.
- Convergence boundary and anchor behavior.
- Journal counting and lineage refresh.
- Complete-file replacement and path guardrails.
- HTTP retries, endpoint modes, JSON/web-search compatibility, and response diagnostics.
- Feature screening and temporal splitting.
- Librarian deterministic/hybrid retrieval and catalog hashes.
- Knowledge-builder staging, replace confirmation, card ranges, truncation, and dropped failures.
- Submission logs, usage capture, manual-intervention summary, and end-to-end orchestration.

In the source environment, focused core, Librarian, end-to-end, submission, and knowledge-builder
tests passed. One full Windows discovery run could not import `tests/test_research.py` because the
active Windows `C:\Python314` interpreter lacked NumPy. This was an environment issue, not a declared
dependency issue: NumPy is present in both `requirements.txt` and `pyproject.toml`, and the user's
normal Linux/WSL Conda environment includes it. Run the full suite again after installing project
dependencies in the fork's active interpreter.

## 21. Known limitations and cautions

- LLM-generated Python runs as ordinary local code inside a filesystem sandbox directory, not a
  hardened OS/container security boundary. Use a container/VM for untrusted model output.
- Token accounting is exact only when the provider reports usage. Command adapters or old HTTP logs
  may yield incomplete totals.
- Cumulative wall-clock is exact only for runs started after `run_timing.json` tracking was added.
- Legacy runs generally lack `predictions_test.npz`; their reporting bundles cannot truthfully
  reconstruct final CSVs without rerunning/reworking the historical model.
- The research screen is a train-only proxy. It can prioritize ideas but does not guarantee official
  validation gains.
- Segment metrics can be noisy for small slices and must not replace official metrics.
- Repeated adaptation to official validation can overfit it; the knowledge base includes proxy and
  confirmation strategies, but the current loop still uses validation for official experiment
  decisions as required.
- The current code scans all eligible historical convergence anchors. In a normal live run it stops
  as soon as the first anchor's third following score is appended.
- Knowledge catalog hashes are canonicalized for CRLF/LF portability. Any manual edit to a live card
  requires updating the corresponding manifest hash or catalog validation will fail.

## 22. Non-negotiable implementation invariants

When continuing work, preserve these unless the user explicitly changes the protocol:

1. Do not modify the Starter Kit or other paths outside `src/parlliament` for ParLLiaMent feature work.
2. Do not let generated code redefine evaluation, labels, validation/test membership, or row order.
3. Do not count screening, planning, retries, or abandoned attempts as experiments.
4. Do not treat parent 0 as measured evidence.
5. Keep code-agent writes restricted to owned files and use complete-file validation.
6. Keep validation checkpoint selection separate from test prediction generation.
7. Never calculate or expose test metrics during research runs.
8. Preserve detailed error/recovery logs even when an attempt later succeeds.
9. Keep segment diagnostics internal; exclude them from submission `iteration_log.json`.
10. Select the globally best validation experiment for submission.
11. Apply the current anchor-based three-following-experiment convergence rule exactly.
12. Preserve Windows/Linux portability and update dependencies when adding non-standard packages.

## 23. Recommended first steps in the forked conversation

1. Read this file, `README.md`, `parlliament_mle_plan.md`, and
   `LIBRARIAN_RETRIEVAL_PLAN.md`.
2. Run `git status --short` and inspect all fork-local changes before editing; do not overwrite
   unrelated user changes.
3. Install the project in the intended environment and run the full test suite.
4. Inspect the latest run with:

   ```bash
   parlliament status src/parlliament/runs/run_12
   ```

5. Open `runs/run_12/submission/results.json`, `iteration_log.json`, and the selected
   `experiment_4` code before proposing further model research.
6. If changing a fixed policy, update implementation, tests, `README.md`,
   `parlliament_mle_plan.md`, the relevant system-owned knowledge card, and its manifest hash
   together.

## 24. Historical decisions worth retaining

- The project began with a fresh simple seed; an optional KuaiRand baseline toggle was added later.
- Additional diagnostic metrics were added so the Judge has richer evidence, but only primary drives
  convergence.
- The train-only screener was introduced because the tight stopping budget makes selective planning
  more valuable than brute-force hypothesis trials.
- Feature schema is always sent to the Feature Engineer for faithfulness.
- Orchestrator instructions are explicit and role-specific.
- Patch-based LLM editing was replaced by validated complete-file replacement after repeated context
  mismatch failures.
- LLM response validation now provides repair feedback for unknown evidence IDs, invalid parents,
  malformed candidate portfolios, and invalid file mappings.
- The Librarian returns final Markdown IDs rather than unnecessary subjective scoring dimensions.
- Knowledge-builder card counts are ranges, web research is available, long cards truncate, and
  exhausted cards drop without aborting the batch.
- Submission reporting was added with per-iteration audit logs, best-validation result deltas,
  resource accounting, and canonical test CSV output.
- Segment diagnostics were explicitly removed from submission `iteration_log.json` while remaining
  available internally.
- The final convergence policy is anchor based: the next three experiments must all miss
  `anchor + 0.002`; submission always uses the global validation best.
