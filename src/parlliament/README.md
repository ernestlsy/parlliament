# ParLLiaMent autonomous recommender MLE system

ParLLiaMent implements the system specified in `parlliament_mle_plan.md`. It is a sequential,
hypothesis-driven experiment engine for KuaiRand-Pure. Generated experiments branch from a scored
parent, change only explicitly delegated files, emit raw validation predictions, and are scored by
evaluation code that agents cannot edit.

## Architecture

The implementation keeps the specification's responsibilities separate:

| Component | Implementation |
|---|---|
| Overseer | `parlliament/overseer.py`: generations, sandboxes, counted IDs, backfill, stopping |
| Research engine | `parlliament/research.py`: leakage policy, profiling, temporal screening, cached evidence |
| Feature Analyst / Evolution Judge / Consultant | `parlliament/agents.py`: evidence review, candidate tournament, selection |
| Librarian | `parlliament/librarian.py`: catalog validation, TF-IDF/MMR retrieval, query expansion, reading-list selection, guarded fetch |
| Literature knowledge base | `parlliament/knowledge/`: categorized Markdown cards, `catalog.jsonl`, and hash manifest |
| Knowledge-base builder | `parlliament/knowledge_builder.py`: manual staged LLM generation, schema repair, hashing, validation, backup, and install |
| Orchestrator | `parlliament/agents.py`: interface contract, selective delegation, targeted debugging |
| Feature Engineer / Model Designer / Trainer | role-specific prompts in `agents.py`, restricted to `data.py`, `model.py`, and `train.py`/`config.json` |
| Experimentor | `parlliament/experimentor.py`: contract probe, bounded subprocess, error classification |
| Journal | `parlliament/journal.py`: durable append-only JSONL archive and convergence checks |
| Guardrails | `parlliament/sandbox.py`: sandbox-contained paths and validated full-file replacements |
| Fixed evaluator | `parlliament/evaluation.py`: official scores plus classification and ranking diagnostics |
| Segment analyzer | `parlliament/diagnostics.py`: post-score warm/cold, context, and activity slices |
| Seed scaffolds | `parlliament/seed/` and `seed_kuairand_baseline/`: selectable unscored parent 0 code |

The default `simple` seed is a fresh, unscored code scaffold—not a prior experiment. It uses only
user/item IDs and a minimal additive pointwise learner. New runs may instead select
`kuairand-baseline`, a ParLLiaMent-owned adaptation of the starter kit's five-field, 16-dimensional
Factorization Machine using user, video, author, tab, and train-fitted duration-bucket fields. Both
choices remain unscored parent 0; published baseline scores are not inserted into the Journal.
Successful descendants are atomically renamed to `runs/<run>/experiment_<id>`. Failed staging directories become
`runs/<run>/abandoned/attempt_<attempt_id>` and retain code and logs without consuming an ID.

## Execution flow

1. Before the first Judge call, the Research Engine profiles the dataset and screens feature groups
   on a temporal holdout entirely inside the April 8-21 training period. It never reads official
   validation rows, writes official predictions, creates a journal record, or consumes an ID.
2. The Feature Analyst interprets the cached screen and latest post-score error slices. The Judge
   may issue up to two rounds of structured research requests. The Librarian combines deterministic
   TF-IDF/MMR retrieval with LLM query expansion, validates a final reading list, and fetches at most
   eight cataloged cards within a 40,000-character generation budget.
3. The informed Evolution Judge creates 12 evidence-citing candidates, the Consultant ranks every
   candidate head-to-head, and the Judge selects exactly one winner. Tournament and literature
   artifacts are retained under `runs/<run>/planning/generation_<n>/` without affecting convergence.
4. Odd generations retain Draft context over the full archive; even generations use the globally
   best scored parent as Improve context. Only the tournament winner proceeds to implementation.
   Eligible parents are recomputed from the durable journal before every initial or replacement call.
5. The Orchestrator receives the deterministic dataset feature schema, fixes the machine-readable
   contract, activates only relevant code agents, and gives each one explicit required changes and
   behavior to preserve.
6. The Feature Engineer always receives the raw-source schema, train-only profiles, leakage status,
   and canonical derived-feature recipes. Each code agent returns the complete contents of every
   file assigned to its role. ParLLiaMent rejects
   absolute/traversing paths, unmanaged or missing files, empty/no-op responses, invalid Python, and
   invalid configuration JSON before replacing anything. Journal diffs are calculated locally.
7. Before training, a small-slice `--contract-check` validates the connected data/model/train path.
   The full process then writes `predictions_valid.npz` containing exactly canonical `row_ids` and
   `scores`. The fixed evaluator independently reloads users and labels from the official validation
   rows, so generated code cannot redefine the ground truth or evaluation population.
8. The fixed Experimentor computes GAUC, nDCG@5, classification diagnostics, and supplementary
   ranking diagnostics. After official scoring succeeds, a non-blocking analyzer adds segment
   diagnostics; only then is the sandbox finalized and the next experiment ID journaled.
   Failures are routed to responsible agents for at most three total attempts and one shared
   wall-clock budget. A repeatedly resource-failing configuration is reclassified as semantic.
9. After every score, ParLLiaMent stops immediately if 50 experiments are counted or if an experiment
   with primary score `x` is followed by three scored experiments that are all strictly below
   `x + 0.002`. This convergence check requires at least four successfully scored experiments.
   The validation-best experiment across the full run, rather than necessarily the final one, is
   selected for submission.

The default backfill ceiling is two replacements per abandoned generation slot. This bounds the
otherwise free abandoned-attempt path while preserving the required 50-count semantics.

## Collected metrics

Every successful journal record keeps the official top-level `GAUC`, `nDCG@5`, and `primary` values.
`primary` remains the only convergence score. Additional nested diagnostics are available to the
Evolution Judge:

- `classification`: accuracy, balanced accuracy, precision, recall, specificity, F1, Matthews
  correlation, predicted-positive rate, the confusion matrix, and the selected threshold. Because
  experiment scores may be logits or arbitrary ranking values, the evaluator selects the threshold
  that maximizes validation F1 instead of assuming a probability cutoff of 0.5. Threshold ties use
  balanced accuracy, accuracy, and then the higher threshold.
- `ranking_diagnostics`: global AUC, average precision, Precision@5, Recall@5, MAP@5, MRR@5, and
  HitRate@5. Per-user top-k metrics are macro-averaged, with zero-positive users contributing zero.
- `data_diagnostics`: label prevalence and prediction-score mean, standard deviation, minimum, and
  maximum. These help detect score collapse and distribution changes.
- `segment_diagnostics`: official-validation error slices by warm/cold user and item, activity,
  request tab, hour, duration, and user positive count. These are retrospective diagnostics, not
  independent test estimates.

The F1-selected classification metrics are diagnostics on the same validation split, so they should
not be treated as unbiased test estimates. The Judge prompt and knowledge base explicitly retain
`primary` as the optimization and stopping objective.

For every Judge proposal and revision call, ParLLiaMent sends a `metric_catalog` describing every field,
the full experiment archive, and a dedicated `scored_metric_history` containing each successful
experiment's complete nested metric object. This makes classification, ranking, and score-distribution
diagnostics directly available for hypothesis decisions without changing the official objective.

## Populate the literature knowledge base once

The population command is a pre-run preparation utility. Run it manually before starting ParLLiaMent;
`parlliament run` never imports it, calls it, or mutates the installed knowledge base. Its default
`extend` mode preserves the current cards and lets the curator choose four to ten non-duplicate
cards for every generated research category:

```bash
parlliament-populate-knowledge \
  --model YOUR_CAPABLE_MODEL \
  --mode extend
```

`OPENAI_API_KEY` supplies the key. The equivalent source script is
`python scripts/populate_knowledge_base.py ...` after installing this package. Provider options
(`--base-url`, `--api-mode`, `--no-json-mode`, `--llm-timeout`, and `--llm-retries`) have the same
meaning as they do for `parlliament run`. A local JSON adapter can instead be supplied with
`--llm-command "your-adapter --json"`.

HTTP-model generation enables OpenAI Responses API hosted web search by default, requires at least
one search on every planning and card-writing request, and uses `high` search context. Every card
must receive at least one URL citation annotation; validated citation titles and URLs are appended
under `Audited web sources`, while search actions and sources remain in the LLM audit. This requires
the Responses API: `--api-mode chat` and compatible providers that resolve to Chat Completions are
rejected unless `--disable-web-search` is supplied. Use `--web-search-context-size low|medium|high`
to change the search budget. A command adapter is responsible for providing and auditing its own web
search because ParLLiaMent cannot inject a hosted provider tool into an arbitrary local command. OpenAI
does not permit hosted web search and provider-enforced JSON mode in the same request, so the builder
automatically omits `text.format` for these requests, explicitly asks for JSON in the prompt, and
validates the returned JSON locally. `--no-json-mode` is therefore not needed for web-enabled builds;
non-search requests continue to use provider-enforced JSON mode by default.

Generation is deliberately staged. For each selected category, the model first proposes catalog
metadata; it then writes each card separately. Invalid schemas, duplicate IDs, near-duplicate topics,
bad Markdown structure, and undersized cards are returned to the model for repair. Cards longer than
the configured maximum are truncated without another LLM call. If a card still cannot be generated
after all response attempts, it is dropped and recorded with its final failure reason while the
remaining cards continue. Only successfully written cards enter later planning context and the final
catalog. The utility writes
the catalog and canonical Windows/Linux hashes into a temporary sibling directory and loads that
directory through the production `KnowledgeCatalog` validator before installation. The prior live
directory is retained as `knowledge.backup-<timestamp>`, and full LLM traffic plus a concise result
report are written beside it.

The repository owns the `task` and `dataset` categories because their facts come from the fixed
specification, evaluator, and supplied CSVs. They are excluded from `--category` and from default LLM
generation. The all-category default therefore makes between 45 and 99 LLM requests (9 category
plans followed by 36-90 card-writing calls), so a targeted first pass is often more practical:

```bash
parlliament-populate-knowledge \
  --model YOUR_CAPABLE_MODEL \
  --category architectures \
  --category training \
  --category experiment_strategy \
  --minimum-cards-per-category 5 \
  --maximum-cards-per-category 10 \
  --guidance-file my_research_priorities.md
```

Use `--mode replace --yes` only when intentionally rebuilding the retrievable research catalog.
System-owned task, dataset, leakage, and evaluation cards are retained even in replacement
mode. Replacement still creates a backup, and generated research and citations should always receive
human review before the backup is removed or the cards guide expensive experiments.

## Install and run

From this directory:

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
parlliament run \
  --workspace . \
  --data-dir ../kuairand-starter-kit/KuaiRand-Pure/data \
  --seed-model simple \
  --model YOUR_MODEL \
  --base-url https://api.openai.com/v1
```

Use `--seed-model kuairand-baseline` on a new run to start parent 0 from the starter kit's official
FM architecture and training setup. The default is `simple`. Seed selection is immutable for a run:
resuming an existing `--run-name` with a different value is rejected to preserve lineage.

`OPENAI_API_KEY` supplies the key. `--base-url` may point to a service implementing the common
chat-completions JSON interface. One client object is shared by all roles. `--api-mode auto` is the
default: it uses the Responses API for `api.openai.com` and Chat Completions for other base URLs.
Override this with `--api-mode responses` or `--api-mode chat`. For a compatible provider that
rejects provider-side JSON mode, add `--no-json-mode`; ParLLiaMent still validates the returned JSON.

HTTP failures include the endpoint, status, request ID when supplied, and the provider's response
body. This makes model-access, unsupported-parameter, and endpoint mismatch errors visible. The
adapter deliberately omits `temperature`, which is not accepted by every current model.
Transient read timeouts, network failures, rate limits, and server errors are retried with bounded
exponential backoff. `--llm-timeout` controls the timeout for each request (default 300 seconds), and
`--llm-retries` controls the number of HTTP retries (default 2).

For a local or custom provider, pass `--llm-command "your-adapter --json"`. The command receives one
JSON object on stdin:

```json
{"role":"consultant","system":"...","payload":{"hypothesis":"..."}}
```

It must print the requested JSON object to stdout. Prompts contain the exact response schema for
each role. This adapter is also useful for audit/replay and local models.

Resume a run with the same `--workspace` and `--run-name`. The append-only journal determines the
next experiment ID, next generation, parent archive, and stop state. Inspect it with:

```bash
parlliament status runs/run_1
```

`status` includes the ten most recent abandoned attempts with their failure stage, exact reason, and
sandbox path. Each new attempt sandbox also contains:

- `attempt_summary.json`: machine-readable outcome, stage, failure chain, elapsed time, and log index.
- `failure.log`: human-readable abandonment reason and full exception traceback.
- `llm_events.jsonl`: complete role-tagged LLM requests, responses, and errors for that attempt.
- `patch_history.json`: versioned complete-file responses, including rejected replacements and
  repair attempts; the historical filename is retained for compatibility.
- `contract_attempt_<n>.log` and `attempt_<n>.log`: contract-probe and training stdout/stderr.

The run directory contains a complete cross-attempt `llm_events.jsonl`, including Judge and
Consultant calls that occur before an experiment sandbox exists. Invalid Orchestrator contracts and
invalid complete-file responses are automatically returned to the responsible agent with precise
validation feedback for up to three response repairs before abandonment.

## Submission bundle

When a run stops, ParLLiaMent writes `runs/<run>/submission/` automatically. Refresh or backfill the
reporting files for any run with:

```bash
parlliament submit runs/run_1
```

The bundle contains:

- `iteration_log.md` and `iteration_log.json`: every scored experiment and abandoned recovery
  attempt, including its hypothesis, final unified code/config diff, validation GAUC and nDCG@5,
  failure chain, and recovery action. Manual interventions are reported as zero/none.
- `kuairand_pure_submission.csv`: predictions from the validation-best experiment in the Starter
  Kit schema `row_id,user_id,video_id,score`, aligned to the canonical test row order.
- `results.md` and `results.json`: the validation-best GAUC, nDCG@5, and primary score, plus absolute
  deltas from the official validation FM baseline (`0.6674`, `0.5357`, and `0.6016`).
- `manifest.json`: bundle completeness, selected experiment, file index, row count, and any reason
  the final CSV could not be produced.

Every successful experiment must emit both `predictions_valid.npz` and `predictions_test.npz` in
one training pass. Checkpoint selection remains validation-only; the Experimentor validates the
test artifact's schema and alignment without calculating or exposing test metrics. Provider-reported
input/output token usage is attached to each new LLM audit event, while `run_timing.json` accumulates
agent wall-clock time across resumed invocations. Runs created before these fields were introduced
can still receive logs and validation result tables, but their token totals or test CSV may be marked
incomplete when the original artifacts do not contain the necessary information.

Important options include `--max-experiments` (default 50), `--timeout` (one wall-clock budget per
experiment), `--max-debug-attempts` (default 3), `--max-backfills` (default 2),
`--candidate-pool-size` (default 12), `--screening-timeout` (default 900 seconds),
`--screening-holdout-fraction` (default 0.25), `--llm-timeout` (default 300 seconds),
`--llm-retries` (default 2), `--literature-rounds` (default 2),
`--literature-max-documents` (default 8), `--literature-character-budget` (default 40000),
`--disable-literature`, and `--force-rescreen`.

The screen writes `runs/<run>/research/feature_catalog.json`, `screening_report.json`, and
`manifest.json`. The manifest records the internal date boundary, dataset fingerprint, dependency
versions, and the fact that no official-validation data or experiment IDs were used. Valid cached
screens are reused when a run resumes.

## Invariants and trust boundary

- Agent output is data: it can only replace the complete contents of its role's allowlisted files.
  All owned files are required and validated together before any replacement is installed.
- Evaluation is imported from the installed ParLLiaMent package, never copied into experiment sandboxes.
- Subprocess arguments are arrays, not shell strings; training executes strictly one experiment at
  a time.
- Config is the shared hyperparameter source. The contract declares mandatory keys and preflight
  rejects omissions.
- Every completed or abandoned attempt is journaled with lineage, scores, diffs, active agents,
  revision count, metrics/failure reason, and sandbox path.
- Research and planning artifacts are deliberately outside the journal and cannot advance the
  experiment counter or the convergence window.
- Literature is fetched only by validated catalog ID. Paths, hashes, candidate-pool membership,
  uniqueness, document count, and the generation character budget are system-enforced; retrieved
  text cannot override fixed task, leakage, evaluation, or file guardrails.
- The host process controls files it writes. Generated Python is trusted to run as ordinary local
  experiment code; use an OS/container sandbox if the LLM itself is not trusted.

## Tests

```bash
python -m unittest discover -s tests -v
```

The suite covers metric parity cases, convergence boundary behavior, journal ID semantics,
replacement/path guardrails, failure classification, contract validation, and a complete mocked LLM
run.
