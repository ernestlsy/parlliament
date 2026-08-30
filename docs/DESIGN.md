# Design — autonomous MLE agent

The implemented system lives in `src/vibes/`. Its own spec is
`src/vibes/agentic_recsys_mle_plan.md`; its user-facing docs are `src/vibes/README.md`.
This file records the architecture at a glance plus the decisions and open design
questions that sit above it.

## Shape of the loop

Sequential, no parallelism. Experiments are grouped into **generations** of 1–3.

```
Overseer ──> Evolution Judge ──> Consultant ──> Orchestrator ──┬─> Feature Engineer  (data.py)
   ^          (hypotheses)       (novelty +     (contract +    ├─> Model Designer    (model.py)
   │                              feasibility)   delegation)   └─> Trainer  (train.py/config.json)
   │                                                                    │
   └──────────── Journal <──── Experimentor (runs it, fixed evaluator) <┘
```

| Component | File | Responsibility |
|---|---|---|
| Overseer | `overseer.py` | Generations, sandboxes, counted IDs, backfill, stopping, shared LLM client |
| Evolution Judge | `agents.py` | Hypotheses; Draft mode (odd gens, 1–3 proposals over full archive) / Improve mode (even gens, 1 proposal from newest generation's best) |
| Consultant | `agents.py` | Novelty + feasibility review against full journal; max 3 revision rounds |
| Orchestrator | `agents.py` | Interface contract, selective sub-agent activation, targeted debug routing |
| Feature Engineer / Model Designer / Trainer | `agents.py` | Patch only their allowlisted file |
| Experimentor | `experimentor.py` | Contract probe, bounded subprocess, error classification |
| Journal | `journal.py` | Append-only JSONL, lineage, convergence checks |
| Guardrails | `sandbox.py` | Path containment, strict single-file unified diffs |
| Fixed evaluator | `evaluation.py` | Official scores + classification/ranking diagnostics. **Agents cannot edit this.** |
| Seed | `seed/` | Neutral two-ID additive learner, unscored "parent 0" |

## Load-bearing invariants

- **Agent output is data.** It can only ever become a patch to that role's allowlisted files.
- **Evaluation is imported from the installed package**, never copied into a sandbox.
  Generated code cannot redefine ground truth or the evaluation population — the
  evaluator independently reloads users and labels from the official validation rows.
- **Config is the single hyperparameter source.** The contract declares mandatory keys;
  preflight rejects omissions.
- **Subprocess args are arrays, not shell strings.** One experiment at a time.
- **Everything is journaled** — completed and abandoned alike, with lineage, diffs,
  active agents, revision count, metrics or failure reason, sandbox path.

## Counting and stopping

- Failed attempts land in `runs/<run>/abandoned/attempt_<hash>/` and consume **no**
  experiment ID. Successful ones are atomically renamed to `experiment_<id>`.
- Backfill ceiling: 2 replacements per abandoned generation slot. Bounds the otherwise
  free abandoned path while preserving 50-count semantics.
- Stop when 50 experiments are counted, **or** when newest primary minus the primary two
  counted experiments earlier is `< 0.002`. Matches the official ε = 0.002 / N = 3.
- `--max-debug-attempts` 3 (mirrors AIDE's `max_debug_depth`), one shared wall-clock
  budget per experiment. A repeatedly resource-failing config is reclassified as semantic.

## Metrics collected

`primary` is the only convergence and optimisation objective. Beyond it the journal
keeps diagnostics that the Judge can read:

- `classification` — accuracy, balanced accuracy, precision, recall, specificity, F1,
  Matthews correlation, predicted-positive rate, confusion matrix, chosen threshold.
  Threshold is picked to maximise validation F1, because experiment scores may be logits
  or arbitrary ranking values rather than probabilities.
- `ranking_diagnostics` — global AUC, average precision, P@5, R@5, MAP@5, MRR@5, HitRate@5.
- `data_diagnostics` — label prevalence, prediction mean/std/min/max. Catches score collapse.

These are diagnostics on the same validation split, not unbiased test estimates. Every
Judge call receives a `metric_catalog`, the full archive, and `scored_metric_history`.

---

## Decisions taken (2026-08-30)

| Decision | Choice |
|---|---|
| Code-agent edit format | **Search/replace blocks.** Replaces unified diffs. |
| `.tw` agents to import | **Research Agent and Data Scientist Agent**, both. |
| 50-iteration cap accounting | **Scored experiments only.** Abandoned attempts consume no slot. |

Rationale and consequences for each are in the sections below.

## Decisions to revisit

### Patch format — DECIDED: search/replace blocks

The agent emits exact `old_string` → `new_string` pairs, applied by literal match:

```
EDIT model.py
<<<<<<< SEARCH
        probabilities = sigmoid(logits)
        gradient = ((probabilities - labels) / size)
=======
        probabilities = softmax_within_user(logits, users)
        gradient = (probabilities - labels / pos_count)
>>>>>>> REPLACE
```

No hunk headers, no line-count arithmetic — the two failure modes that killed every
run_2 attempt. Keeps the fine-grained auditability that whole-file rewrite loses, and
costs fewer output tokens than a rewrite (wall-clock and tokens are both scored).

Implementation notes for whoever builds it:

- Guardrails in `sandbox.py` stay as they are — path containment and the per-role file
  allowlist are orthogonal to the edit format and are load-bearing.
- Reject a non-unique `SEARCH` match rather than applying the first hit. Ambiguity is a
  contract failure and should route back to the agent like any other.
- Keep the 3-response repair loop; feed the exact rejection reason back.
- The Journal must still store a **computed** unified diff vs parent — deliverable 3
  asks for "the code diff applied", and that can be derived after the fact. The
  agent-facing edit format and the logged format do not have to match.

### Original options considered

Unified diffs are the current interface between code agents and the sandbox, and they
are where the system is bleeding. See `docs/STATUS.md` "Failure 1": **100% of run_2's
abandonments were `patch context does not match reference file`**, with correct code
inside the rejected patch. LLMs are unreliable at emitting exact hunk headers and
context lines; the surrounding machinery is fine.

Options, roughly in order of expected payoff per unit of work:

1. **Search/replace blocks** — agent emits exact `old_string` → `new_string` pairs,
   applied by literal match. Same auditability as a diff, far less to get wrong. This is
   what production coding agents converged on.
2. **Whole-file rewrite against the interface contract** — simplest, most robust,
   loses fine-grained diff auditability. The Journal can still store a *computed* diff,
   which is what deliverable 3 actually asks for.
3. **Fuzzy/context-tolerant patch application** — retry with whitespace-insensitive and
   offset-tolerant matching before giving up.

Note that the diff-vs-parent required by deliverable 3 can always be *computed* after
the fact. Nothing forces the agent-facing edit format to be a unified diff.

### Seed initialisation

Zero-init makes every multiplicative interaction term inert (`docs/STATUS.md`
"Failure 2"). Fix the seed, and add a smoke check that parameter blocks actually move
during training, so a dead-gradient experiment fails loudly instead of silently
reporting the parent's score.

### Ideas from `src/.tw/agents.md` — DECIDED: import both

tw's design covers the same loop with 8 agents. Two of them have no counterpart in the
current implementation and map directly onto judged criteria. **Both are in scope**,
but sequence them after the patch and seed fixes — they add LLM calls and wall-clock,
and Feasibility is only scored once we clear the baseline.

- **Research Agent** — searches papers, public implementations, industry practice, and
  converts them into actionable hypotheses with evidence, expected benefit, risk, and
  complexity. Innovation & Problem Insight (20%) explicitly rewards "originality in
  drawing on published methods... beyond naive baseline tweaks". Today this is a static
  `knowledge/recommender_research.md`; a live research step is strictly more than that.
- **Data Scientist Agent** — EDA before hypothesising: cardinality, label imbalance,
  temporal patterns, duplicate interactions, leakage risk. The current system has no
  data-inspection stage at all, and Figure 1 of the problem statement lists
  "inspect data" as a core loop stage.

tw also proposes a **Persistent Judge** (research memory) separate from the Evolution
Judge (what to try next). Our Journal already fills that role; the split is a naming
difference, not a missing capability.

tw's **unified experiment specification** — one YAML-ish object carrying
`hypothesis` + `research` + `data_findings` + `features` + `model` + `training` — is a
tidier contract than the current split between hypothesis text and interface contract,
and it maps cleanly onto the per-iteration log deliverable.

Two of tw's judgement calls are worth keeping as-is in our system: **no separate
Evaluation Agent** (evaluation is deterministic, don't spend an LLM call on it) and
**no separate mutation/crossover agents** (those are operations of the Judge and the
code agents, not roles).

### Where the science should go first

Once patching is fixed, the organizers' unexplored list (`docs/BENCHMARK.md` §6) is the
map: ranking losses first (BPR / within-user listwise softmax), then user history
sequences, then multi-task auxiliary heads, then censored watch-time. Architecture
swaps rank *below* all of those — capacity is measured not to be the bottleneck.
The Judge already proposes exactly these; it just needs them to survive to execution.
