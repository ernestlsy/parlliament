# Agentic MLE System for Recommender Systems — Revised Plan

## 1. Overview

An evolutionary, hypothesis-driven agentic system that autonomously trains and improves a recommender-system model. The system runs sequentially (no parallelism), is capped at **50 counted experiments**, and stops on either the 50-cap or a convergence rule based on validation score plateauing.

Experiments are grouped into **generations** (1-3 experiments each). Even-numbered generations use *Improve mode*, odd-numbered generations use *Draft mode*. Crashed/abandoned attempts do **not** consume an experiment ID or count toward the 50-cap, but are bounded by a retry ceiling so they don't consume unlimited time.

---

## 2. External Components (cross-generation workflow)

### 2.1 Overseer
- Initializes and spawns the internal agent set for each experiment.
- Creates a sandbox directory per experiment: `runs/run_1/experiment_<id>/` — all generated code and logs for that experiment live here only.
- Holds the shared LLM client instance used by all agents.
- Checks the convergence rule after every successfully scored experiment.
- Enforces the guardrail: agents may only write files inside their own experiment's sandbox directory.
- Runs strictly sequentially — one experiment at a time, no concurrent sandboxes.

### 2.2 Evolution Judge (LLM agent)
Analyzes literature and prior performance to propose new hypotheses. Has access to a knowledge base of literature/research.

Before candidate generation, the Judge may emit structured research requests. A Librarian combines
catalog filters, TF-IDF/MMR retrieval, LLM query expansion, and a validated ID-only reading list;
system code then fetches bounded Markdown context for a second, informed Judge call.

**a. Improve mode** (even-numbered generations)
- Sees hypothesis, config, code, and performance of **one** experiment from the most recent generation.
- Proposes **1** improvement hypothesis referencing that experiment.

**b. Draft mode** (odd-numbered generations)
- Sees hypotheses, configs, code, and performance of **all experiments in the full archive** (not just the latest generation — see §5.4), so genuinely good ideas from earlier generations aren't permanently lost.
- Proposes **1-3** new hypotheses, each referencing exactly one prior experiment.
- Reference selection is a **mix of top-scoring and diverse/under-explored** prior experiments, not purely greedy best-first, to avoid premature convergence to a local optimum.

**Novelty check:** before finalizing hypotheses, the Judge checks them against the full hypothesis history (via the Consultant, see below) to avoid re-proposing something structurally equivalent to an already-tested or already-failed idea.

**Structured hypothesis output:** each hypothesis is scored by the Judge on *interestingness*, *novelty*, and *feasibility* (1-10 scale) alongside the free-text description, giving the Consultant and Journal an auditable, comparable rubric instead of free-form judgment.

### 2.3 Consultant (LLM agent)
- Reviews Evolution Judge's suggestions against the **full history** of experiments (all generations) for feasibility and likely effectiveness.
- Rejects duplicate/near-duplicate hypotheses (novelty check) and structurally infeasible ones.
- If rejected, returns feedback to the Evolution Judge for revision.
- **Capped revision loop:** maximum 3 revision rounds per hypothesis slot. If still rejected after 3 rounds, the Consultant either (a) accepts the best of the 3 attempts with a flagged caveat, or (b) the slot is dropped for that generation (generation runs with fewer than its planned experiment count). No infinite loop.

### 2.4 Journal
Logs every completed and abandoned attempt. Schema per record:

| Field | Description |
|---|---|
| `experiment_id` | Numeric ID, 1-onwards, assigned only on a **counted** (non-crashed) experiment |
| `generation` | Generation number |
| `parent_experiment_id` | Reference experiment the hypothesis was derived from |
| `hypothesis_text` | Free-text hypothesis |
| `hypothesis_scores` | Interestingness / novelty / feasibility (1-10) |
| `mode` | improve / draft |
| `code_diff` | Diff applied vs. parent's code (per file) |
| `config_diff` | Diff applied vs. parent's config |
| `active_sub_agents` | Which of Feature Engineer / Model Designer / Trainer actually ran (see §5.2) |
| `metrics` | Final validation metrics (NDCG@K, Recall@K, etc.) |
| `status` | scored / abandoned |
| `failure_reason` | Populated only if abandoned — used by Evolution Judge to avoid repeat failures |
| `consultant_rounds` | Number of revision rounds taken |

Abandoned attempts are logged but excluded from the 50-count and from convergence scoring.

---

## 3. Internal Components (per-experiment ML loop)

### 3.1 Orchestrator (LLM agent)
- Receives a single hypothesis + the reference experiment's code/config.
- Determines an **interface contract** for this experiment before delegating: expected schema of `data.py`'s output, config keys touched, and `model.py`/`train.py`'s expected inputs. This travels with the sub-agent prompts.
- Decides which sub-agent(s) are **active** for this hypothesis — e.g., a loss-function change only activates Model Designer; Feature Engineer's and Trainer's files are copied forward unchanged from the reference experiment.
- Crafts each active sub-agent's prompt, instructing it to return the **complete final content of every file owned by that role**. Ernest validates these files before installation and computes audit diffs itself.
- Owns the retry loop for this experiment (see §4) — classifies Experimentor failures and re-routes to the correct sub-agent(s).

### 3.2 Feature Engineer (LLM agent, writes code)
- Generates/patches `data.py`.
- Sees `data.py` from the reference experiment and the interface contract it must satisfy.
- Only invoked when active for the current hypothesis.

### 3.3 Model Designer (LLM agent, writes code)
- Generates/patches `model.py`, including loss function design.
- Sees `model.py` from the reference experiment and the interface contract.
- Only invoked when active for the current hypothesis.

### 3.4 Trainer (LLM agent, writes code)
- Generates/patches `train.py`, including training hyperparameters/config.
- Sees `train.py` from the reference experiment and the interface contract.
- Only invoked when active for the current hypothesis.

### 3.5 Experimentor
- Runs the (possibly partially regenerated, partially carried-forward) pipeline files.
- **Evaluation code is hard-coded** and never modified by any agent.
- On failure: classifies the error (see §4), packages a structured error report, and returns it to the Orchestrator instead of retrying blindly itself.
- On success: records results/metrics to the Journal and assigns the experiment its numeric ID.

---

## 4. Retry & Failure Handling

**Principle:** the Orchestrator owns retry decisions; sub-agents only ever see a structured error report and propose a fix — they never decide whether to keep retrying.

**Step 1 — Classify the failure** (Experimentor):
- *Semantic/logic error* — exception, shape mismatch, NaN loss → responsible sub-agent's bug, feed traceback back, retry.
- *Contract violation* — one file's output doesn't match another's expected input schema → route to Orchestrator, since it may need coordinated multi-file changes.
- *Resource/transient error* — OOM, timeout → retry once with reduced batch size/timeout; if it recurs, reclassify as semantic (the config is genuinely wrong) rather than retrying indefinitely.

**Step 2 — Targeted routing.** Orchestrator sends the structured error to only the sub-agent(s) responsible for the offending file(s), along with that agent's own prior code — not a full-context rewrite of everything.

**Step 3 — Bounded retries.** Cap at **3 debug attempts per experiment** (mirrors AIDE's `max_debug_depth` pattern). Each attempt includes the fresh traceback so it's a genuine self-correction step, not a blind repeat.

**Step 4 — Abandon and backfill on exhaustion.** After 3 failed attempts (or a wall-clock ceiling, whichever comes first — e.g. N minutes), the experiment is abandoned:
- Does **not** consume one of the 50 experiment slots.
- Logged in the Journal as `status: abandoned` with `failure_reason`.
- Evolution Judge generates one replacement hypothesis to refill the generation's slot, informed by the failure reason so it avoids the same structural issue.
- The replacement gets a fresh, never-reused experiment ID once it succeeds.

**Step 5 — Hard time backstop.** Because abandoned attempts are free with respect to the 50-cap, add an explicit per-experiment wall-clock ceiling regardless of retry count, so a single stubborn bug cannot silently consume a large share of total hackathon time while "not counting" against the budget.

---

## 5. Design Details Carried Over from Discussion

### 5.1 Interface Contracts
Before delegation, the Orchestrator defines (and the Experimentor validates on a small data slice before a full run):
- `data.py` output schema (column names, dtypes, feature dimensions).
- Shared config object — the single source of truth for hyperparameters; no sub-agent hardcodes its own values outside it.
- `model.py` → `train.py` expected tensor/interface shapes.

Validation failures are classified as *Contract Fulfillment Failures* (producer didn't meet spec) or *Contract Usage Violations* (consumer misused input), which determines whether the fix routes to the producer or consumer sub-agent.

### 5.2 Selective Delegation
Not every hypothesis requires all three sub-agents to regenerate code. The Orchestrator explicitly marks which sub-agents are active; inactive ones' files are copied forward unchanged from the reference experiment, reducing both token cost and regression risk.

### 5.3 Validated Full-File Code Edits
Sub-agents return complete files rather than authoring fragile diff metadata. Ernest requires exactly
the role's allowlisted files, validates Python syntax and configuration JSON, installs all files as
one rollback-protected operation, and computes parent-relative diffs locally for the journal.

### 5.4 Full-Archive Reference (not just latest generation)
Draft mode's reference pool spans the entire experiment history, not just the immediately preceding generation, so a promising idea that lost out in an earlier generation can still be revived later.

### 5.5 Diversity in Reference Selection
The Judge deliberately mixes top-scoring and under-explored prior experiments as hypothesis references, rather than always branching from the single best-performing one, to reduce risk of local-optimum convergence.

### 5.6 Novelty / Duplicate Check
The Consultant cross-checks new hypotheses against the full hypothesis history (including abandoned ones) to avoid re-testing ideas that already failed or already succeeded in a materially identical form.

---

## 6. Convergence & Stopping Rule (Hackathon-Constrained)

Because experiments run strictly sequentially and are numbered 1-onwards in execution order, "3 consecutive experiments" is unambiguous:

1. Maintain a running ordered list of `(experiment_id, val_score)` for **successfully scored** experiments only (abandoned attempts never enter this list).
2. After each new successful experiment, compare its score against the scores 1 and 2 positions back in this list.
3. If the score has increased by **less than 0.002** across these 3 consecutive entries, **stop** — this is the converged score used for judging.
4. Independently, stop if the counted-experiment total reaches **50**, whichever comes first.
5. The judged score is the converged score at the stopping point, not the peak or final score reached afterward — no further experiments should run once either condition triggers.

---

## 7. Component Responsibility Summary

| Component | Responsibility |
|---|---|
| Overseer | Sandbox creation, LLM client, sequential experiment spawning, convergence checks, guardrail enforcement |
| Evolution Judge | Hypothesis generation (improve/draft), structured scoring, novelty filtering, diverse reference selection |
| Librarian | Catalog filtering, hybrid literature retrieval, ID-only selection, guarded Markdown fetch, retrieval audit |
| Consultant | Feasibility/novelty sanity check, capped revision loop (max 3 rounds), full-history awareness |
| Journal | Full experiment + abandoned-attempt logging, lineage tracking, hypothesis scores |
| Orchestrator | Interface contract definition, selective sub-agent activation, full-file delegation, retry ownership, error routing |
| Feature Engineer / Model Designer / Trainer | Return complete content for their respective allowlisted files, only when active for the hypothesis |
| Experimentor | Runs pipeline, hard-coded evaluation, failure classification, metric logging |

---

## 8. Explicit Non-Requirements (per current constraints)

- **No parallelism**: experiments run one at a time; Overseer does not need concurrent sandbox/GPU allocation logic.
- **Crashes excluded from the 50-cap**: only successfully scored experiments consume an ID; retries and abandonments are free with respect to the cap but bounded by retry count and wall-clock ceiling.
- **Convergence rule is fixed** by hackathon requirements and is not to be replaced by a peak- or final-score metric — the system must optimize for and report the converged score specifically.
