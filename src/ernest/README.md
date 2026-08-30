# Ernest autonomous recommender MLE system

Ernest implements the system specified in `agentic_recsys_mle_plan.md`. It is a sequential,
hypothesis-driven experiment engine for KuaiRand-Pure. Generated experiments branch from a scored
parent, change only explicitly delegated files, emit raw validation predictions, and are scored by
evaluation code that agents cannot edit.

## Architecture

The implementation keeps the specification's responsibilities separate:

| Component | Implementation |
|---|---|
| Overseer | `agentic_recsys/overseer.py`: generations, sandboxes, counted IDs, backfill, stopping |
| Research engine | `agentic_recsys/research.py`: leakage policy, profiling, temporal screening, cached evidence |
| Feature Analyst / Evolution Judge / Consultant | `agentic_recsys/agents.py`: evidence review, candidate tournament, selection |
| Literature knowledge base | `agentic_recsys/knowledge/`: read-only research and known-result context for the Judge |
| Orchestrator | `agentic_recsys/agents.py`: interface contract, selective delegation, targeted debugging |
| Feature Engineer / Model Designer / Trainer | role-specific prompts in `agents.py`, restricted to `data.py`, `model.py`, and `train.py`/`config.json` |
| Experimentor | `agentic_recsys/experimentor.py`: contract probe, bounded subprocess, error classification |
| Journal | `agentic_recsys/journal.py`: durable append-only JSONL archive and convergence checks |
| Guardrails | `agentic_recsys/sandbox.py`: sandbox-contained paths and strict single-file unified diffs |
| Fixed evaluator | `agentic_recsys/evaluation.py`: official scores plus classification and ranking diagnostics |
| Segment analyzer | `agentic_recsys/diagnostics.py`: post-score warm/cold, context, and activity slices |
| Seed scaffold | `agentic_recsys/seed/`: neutral two-ID additive learner, represented as unscored parent 0 |

The seed is a fresh, unscored code scaffold—not a prior experiment and not a reference to the
published KuaiRand baseline. It uses only user/item IDs and a minimal additive pointwise learner so
the first generation starts without inherited interaction architecture or performance assumptions.
Successful descendants are atomically renamed to `runs/<run>/experiment_<id>`. Failed staging directories become
`runs/<run>/abandoned/attempt_<attempt_id>` and retain code and logs without consuming an ID.

## Execution flow

1. Before the first Judge call, the Research Engine profiles the dataset and screens feature groups
   on a temporal holdout entirely inside the April 8-21 training period. It never reads official
   validation rows, writes official predictions, creates a journal record, or consumes an ID.
2. The Feature Analyst interprets the cached screen and the latest post-score error slices. The
   Evolution Judge creates 12 evidence-citing candidates, the Consultant ranks every candidate
   head-to-head, and the Judge selects exactly one winner. The full tournament is retained under
   `runs/<run>/planning/generation_<n>/` without affecting convergence.
3. Odd generations retain Draft context over the full archive; even generations use the globally
   best scored parent as Improve context. Only the tournament winner proceeds to implementation.
   Eligible parents are recomputed from the durable journal before every initial or replacement call.
4. The Orchestrator fixes the machine-readable contract and activates only relevant code agents.
5. Each code agent returns unified diffs. Ernest rejects absolute/traversing paths, unmanaged files,
   mismatched patch headers, and stale patch context.
6. Before training, a small-slice `--contract-check` validates the connected data/model/train path.
   The full process then writes `predictions_valid.npz` containing exactly canonical `row_ids` and
   `scores`. The fixed evaluator independently reloads users and labels from the official validation
   rows, so generated code cannot redefine the ground truth or evaluation population.
7. The fixed Experimentor computes GAUC, nDCG@5, classification diagnostics, and supplementary
   ranking diagnostics. After official scoring succeeds, a non-blocking analyzer adds segment
   diagnostics; only then is the sandbox finalized and the next experiment ID journaled.
   Failures are routed to responsible agents for at most three total attempts and one shared
   wall-clock budget. A repeatedly resource-failing configuration is reclassified as semantic.
8. After every score, Ernest stops immediately if 50 experiments are counted or if the newest score
   minus the score two counted experiments earlier is strictly less than `0.002`.

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

For every Judge proposal and revision call, Ernest sends a `metric_catalog` describing every field,
the full experiment archive, and a dedicated `scored_metric_history` containing each successful
experiment's complete nested metric object. This makes classification, ranking, and score-distribution
diagnostics directly available for hypothesis decisions without changing the official objective.

## Install and run

From this directory:

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
ernest run \
  --workspace . \
  --data-dir ../kuairand-starter-kit/KuaiRand-Pure/data \
  --model YOUR_MODEL \
  --base-url https://api.openai.com/v1
```

`OPENAI_API_KEY` supplies the key. `--base-url` may point to a service implementing the common
chat-completions JSON interface. One client object is shared by all roles. `--api-mode auto` is the
default: it uses the Responses API for `api.openai.com` and Chat Completions for other base URLs.
Override this with `--api-mode responses` or `--api-mode chat`. For a compatible provider that
rejects provider-side JSON mode, add `--no-json-mode`; Ernest still validates the returned JSON.

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
ernest status runs/run_1
```

`status` includes the ten most recent abandoned attempts with their failure stage, exact reason, and
sandbox path. Each new attempt sandbox also contains:

- `attempt_summary.json`: machine-readable outcome, stage, failure chain, elapsed time, and log index.
- `failure.log`: human-readable abandonment reason and full exception traceback.
- `llm_events.jsonl`: complete role-tagged LLM requests, responses, and errors for that attempt.
- `patch_history.json`: every raw patch response, including rejected patches and repair attempts.
- `contract_attempt_<n>.log` and `attempt_<n>.log`: contract-probe and training stdout/stderr.

The run directory contains a complete cross-attempt `llm_events.jsonl`, including Judge and
Consultant calls that occur before an experiment sandbox exists. Invalid Orchestrator contracts and
malformed code patches are automatically returned to the responsible agent for up to three response
repairs before abandonment.

Important options include `--max-experiments` (default 50), `--timeout` (one wall-clock budget per
experiment), `--max-debug-attempts` (default 3), `--max-backfills` (default 2),
`--candidate-pool-size` (default 12), `--screening-timeout` (default 900 seconds),
`--screening-holdout-fraction` (default 0.25), `--llm-timeout` (default 300 seconds),
`--llm-retries` (default 2), and `--force-rescreen`.

The screen writes `runs/<run>/research/feature_catalog.json`, `screening_report.json`, and
`manifest.json`. The manifest records the internal date boundary, dataset fingerprint, dependency
versions, and the fact that no official-validation data or experiment IDs were used. Valid cached
screens are reused when a run resumes.

## Invariants and trust boundary

- Agent output is data: it can only become a patch to its role's allowlisted files.
- Evaluation is imported from the installed Ernest package, never copied into experiment sandboxes.
- Subprocess arguments are arrays, not shell strings; training executes strictly one experiment at
  a time.
- Config is the shared hyperparameter source. The contract declares mandatory keys and preflight
  rejects omissions.
- Every completed or abandoned attempt is journaled with lineage, scores, diffs, active agents,
  revision count, metrics/failure reason, and sandbox path.
- Research and planning artifacts are deliberately outside the journal and cannot advance the
  experiment counter or the convergence window.
- The host process controls files it writes. Generated Python is trusted to run as ordinary local
  experiment code; use an OS/container sandbox if the LLM itself is not trusted.

## Tests

```bash
python -m unittest discover -s tests -v
```

The suite covers metric parity cases, convergence boundary behavior, journal ID semantics,
diff/path guardrails, failure classification, contract validation, and a complete mocked LLM run.
