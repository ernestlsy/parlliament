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
| Evolution Judge / Consultant | `agentic_recsys/agents.py`: full-history proposals, scoring, novelty review, capped revision |
| Literature knowledge base | `agentic_recsys/knowledge/`: read-only research and known-result context for the Judge |
| Orchestrator | `agentic_recsys/agents.py`: interface contract, selective delegation, targeted debugging |
| Feature Engineer / Model Designer / Trainer | role-specific prompts in `agents.py`, restricted to `data.py`, `model.py`, and `train.py`/`config.json` |
| Experimentor | `agentic_recsys/experimentor.py`: contract probe, bounded subprocess, error classification |
| Journal | `agentic_recsys/journal.py`: durable append-only JSONL archive and convergence checks |
| Guardrails | `agentic_recsys/sandbox.py`: sandbox-contained paths and strict single-file unified diffs |
| Fixed evaluator | `agentic_recsys/evaluation.py`: exact starter-kit GAUC, nDCG@5, and primary score |
| Seed experiment | `agentic_recsys/seed/`: standalone five-field NumPy FM, represented as experiment 0 |

The seed is an uncounted parent with the published validation score. Successful descendants are
atomically renamed to `runs/<run>/experiment_<id>`. Failed staging directories become
`runs/<run>/abandoned/attempt_<attempt_id>` and retain code and logs without consuming an ID.

## Execution flow

1. Odd generations run Draft mode (one to three proposals over the full scored archive); even
   generations run Improve mode (one proposal from the best experiment in the newest scored
   generation).
2. The Consultant checks every proposal against the full journal. Revisions are capped at three.
3. The Orchestrator fixes the machine-readable contract and activates only relevant code agents.
4. Each code agent returns unified diffs. Ernest rejects absolute/traversing paths, unmanaged files,
   mismatched patch headers, and stale patch context.
5. Before training, a small-slice `--contract-check` validates the connected data/model/train path.
   The full process then writes `predictions_valid.npz` containing exactly canonical `row_ids` and
   `scores`. The fixed evaluator independently reloads users and labels from the official validation
   rows, so generated code cannot redefine the ground truth or evaluation population.
6. The fixed Experimentor computes GAUC and nDCG@5 and assigns the next ID only after scoring.
   Failures are routed to responsible agents for at most three total attempts and one shared
   wall-clock budget. A repeatedly resource-failing configuration is reclassified as semantic.
7. After every score, Ernest stops immediately if 50 experiments are counted or if the newest score
   minus the score two counted experiments earlier is strictly less than `0.002`.

The default backfill ceiling is two replacements per abandoned generation slot. This bounds the
otherwise free abandoned-attempt path while preserving the required 50-count semantics.

## Install and run

From this directory:

```bash
python -m pip install -e .
ernest run \
  --workspace . \
  --data-dir ../kuairand-starter-kit/KuaiRand-Pure/data \
  --model YOUR_MODEL \
  --base-url https://api.openai.com/v1
```

`OPENAI_API_KEY` supplies the key. `--base-url` may point to a service implementing the common
chat-completions JSON interface. One client object is shared by all roles.

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

Important options include `--max-experiments` (default 50), `--timeout` (one wall-clock budget per
experiment), `--max-debug-attempts` (default 3), and `--max-backfills` (default 2).

## Invariants and trust boundary

- Agent output is data: it can only become a patch to its role's allowlisted files.
- Evaluation is imported from the installed Ernest package, never copied into experiment sandboxes.
- Subprocess arguments are arrays, not shell strings; training executes strictly one experiment at
  a time.
- Config is the shared hyperparameter source. The contract declares mandatory keys and preflight
  rejects omissions.
- Every completed or abandoned attempt is journaled with lineage, scores, diffs, active agents,
  revision count, metrics/failure reason, and sandbox path.
- The host process controls files it writes. Generated Python is trusted to run as ordinary local
  experiment code; use an OS/container sandbox if the LLM itself is not trusted.

## Tests

```bash
python -m unittest discover -s tests -v
```

The suite covers metric parity cases, convergence boundary behavior, journal ID semantics,
diff/path guardrails, failure classification, contract validation, and a complete mocked LLM run.
