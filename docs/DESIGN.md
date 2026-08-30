# Design — autonomous MLE agent

The implemented system lives in `src/vibes/`. Its own spec is
`src/vibes/agentic_recsys_mle_plan.md`; its user-facing docs are `src/vibes/README.md`.
This file records the architecture at a glance plus the decisions and open design
questions that sit above it.

## Target architecture — island model

This is the team's plan as of 2026-08-30. **The code in `src/vibes/` does not implement
it yet** — it runs a single sequential lineage with Draft/Improve generations (see
"Shape of the loop" below). Treat this section as the target, the next as the present.

### The plan

- Start from **N islands**, each a deliberately dissimilar subset of the candidate
  solution space. Islands exist to inject variability and stop the population collapsing
  into a local minimum too early — they are not themselves discovered.
- Island definitions are **hardcoded**, a **small** fixed set, and pitched at the
  recommender-systems domain rather than at KuaiRand specifically, so the same set is
  **reusable across datasets and runs**.
- A **master agent** runs between iterations. It reads each island's evaluation metrics,
  forms hypotheses, decides which models survive (i.e. which hypotheses worked), and
  generates the next round's hypotheses from the survivors.
- The master also consumes **external documents via RAG or web search**. This is the
  "research" aspect — it maps onto the Research Agent imported from `src/.tw/`.
- Population size stays **flat or grows only slightly**, to bound compute.
- Once a **threshold** is reached, a final hypothesis is formed and the agent builds the
  best model it can from it.

### Mapping onto what already exists

| Plan element | Existing component |
|---|---|
| Master agent | Evolution Judge + Consultant (`agents.py`) |
| Survivor selection | Currently implicit in parent selection — needs an explicit keep/kill step |
| Research via RAG / web | Research Agent (decided import); today only static `knowledge/*.md` |
| Per-island refinement | Orchestrator + Feature Engineer / Model Designer / Trainer |
| Population memory | Journal (`journal.py`) — already stores lineage and `parent_experiment_id` |
| Islands | **No counterpart.** The nearest thing is plan §5.5 "mix top-scoring and under-explored parents", which is diversity heuristics on a single population, not island structure. |

### Budget arithmetic

The cap is 50 **scored** experiments. At one experiment per island per round:

| Islands | Rounds available |
|---|---|
| 3 | 16 |
| 4 | 12 |
| 5 | 10 |
| 6 | 8 |
| 8 | 6 |

Wall-clock is the looser constraint: 6h / 50 ≈ 7 min per experiment, against ~40s for
the baseline FM. Islands are cheap in time, expensive in **iterations**.

### Four conflicts to resolve before building

**1. The convergence rule will kill an exploratory run. Fix first.**

`journal.py:57` implements:

```python
return recent[-1] - recent[0] < epsilon    # raw latest minus raw oldest
```

This compares the *most recent raw score*, not the running best. Any experiment that
scores worse than one three back triggers "converged" and stops the run. Exploration
inherently produces duds, and an island model produces them by design.

This already happened. run_2 scored 0.5885 → 0.5885 → 0.5769; the third is
0.0116 *below* the first, which satisfies `< 0.002`, and the run halted at 3 of 50
scored experiments.

The official wording is "validation primary has not **improved** by more than ε over the
last N = 3 consecutive iterations". Tracking the **running best** satisfies that reading,
is monotone non-decreasing, and is robust to interleaved islands and to duds:

```python
best_now - best_three_iterations_ago < epsilon
```

**2. A post-threshold "build the best model" phase is not scored.**

The rules say the scored submission is "the validation-best checkpoint at [the
convergence] point", and that "no further experiments should run once either condition
triggers". A final model built *after* the stop condition fires is outside the scored
window.

The final-hypothesis build must therefore happen **inside** the 50 and **before** the
stop fires — reserve the last rounds for it rather than treating it as an epilogue.

**3. Hardcoded islands need an axis that survives being hardcoded.**

Since the set is fixed, small, and meant to be reused across datasets, the splitting axis
has to be domain-general for recommender systems — not tuned to KuaiRand-Pure. Two
candidate axes, and only one of them holds up:

- **By model architecture** (FM / DeepFM / DCN / xDeepFM). Generalizes fine, but the
  organizers measured this as the dead axis on this benchmark: embedding capacity
  k = 8/16/32 is flat (0.5895 / 0.5902 / 0.5887), added static features do nothing, and
  they rank "change the model" **fifth**, below losses, sequences, multi-task and
  watch-time (`docs/BENCHMARK.md` §6). Islands split this way would spend the whole
  diversity budget where there is least headroom.
- **By learning objective and signal consumed.** Equally domain-general — every
  recommender ranking problem has a choice of objective and a choice of which feedback
  signals to exploit — and it lines up with the organizers' priority order on this
  benchmark. This is the recommendation.

Architecture then becomes something an island *varies while refining*, not something an
island *is*.

### Proposed island set

Five candidates, each recsys-general and reusable, each mapping to a published family:

| Island | Invariant it must preserve | Maps to |
|---|---|---|
| **Baseline anchor** | Pointwise BCE on ID + categorical fields | The official FM. Doubles as the fix for conflict 4. |
| **Ranking objective** | Loss is pairwise or listwise, never pointwise | BPR, within-user softmax. Organizer priority #1. |
| **Sequence / interest** | Consumes user behaviour history | DIN, SIM, GRU4Rec, SASRec. Priority #2. |
| **Multi-task** | Trains auxiliary feedback heads alongside `long_view` | ESMM, MMoE, PLE. Priority #3. |
| **Bias / watch-time** | Models duration or exposure bias explicitly | CWM censored regression, IPS on the randomized-exposure log. Priorities #4 and #7. |

Start with three or four, not five. At 50 scored experiments, five islands leaves ten
each — thin, given we are still below baseline.

**Islands are not mutually exclusive**, which is the failure mode to guard against: a
sequence model can also adopt a listwise loss, and if every island adopts every trick the
population collapses back to one point and the islands stop buying anything. Define an
island by the **one ingredient it must keep**, and leave everything else free to vary.
The master enforces the invariant at the keep/kill step.

**Two things fall out of hardcoding them.** First, island definitions become declarative
config — name, invariant, seed scaffold, prompt fragment, reference literature — not
code, which is what makes them portable to KuaiRand-1k/27k or another dataset without a
rewrite. Second, RAG gets a natural partition: retrieval is namespaced per island, so the
Research Agent pulls sequence-modelling papers for the sequence island and debiasing
papers for the bias island, instead of one undifferentiated corpus. Cheaper and sharper.

**Note for the report:** hardcoded island priors are a *design-time* choice, not a
run-time intervention, so they do not count against the manual-intervention measure that
Autonomy (20%) is scored on. Say so explicitly, so judges read them as architecture
rather than as hand-holding.

**Convergence interacts badly with islands.** With K islands round-robin, three
consecutive *experiments* span less than one full sweep, so a global N = 3 window fires
almost immediately once islands start exploring. Evaluate convergence on the global
**running best over the last 3 rounds**, not the last 3 experiments — a round being one
sweep across the live islands. Keep iteration = experiment for the 50 cap (conservative
reading, matches the problem statement's own "100 iterations ≈ 28 min" framing), and
document the window interpretation in the report.

**4. Task requirement 1 — reproduce the official baseline — is currently unmet.**

The problem statement's first task requirement is to stand up a pipeline and confirm it
reaches the official baseline's reported validation score (0.6016). `src/vibes/README.md`
states the opposite as a deliberate choice: the seed is "not a reference to the published
KuaiRand baseline", starting instead from a neutral additive learner.

The neutral seed is defensible for avoiding inherited assumptions, but the requirement is
graded. Reproducing the FM baseline as one island — or as an explicit pre-flight step —
satisfies it and simultaneously gives every other island a real reference point instead
of a 0.5885 degenerate one.

### Island lifecycle

**Extinction — decided.** An island is killed when it has **stalled** (no improvement over
its own last N rounds) **and** is **outperformed** by the live leaders. Both conditions,
not either: a level comparison alone would kill a slow island that is still climbing.

Three qualifications:

- **Grace period.** Islands have unequal setup costs. The sequence and multi-task islands
  spend their first experiments building machinery that scores badly before it pays off;
  the FM anchor is at full strength immediately. Extinction on score without a
  minimum-rounds floor systematically executes the high-ceiling islands — greedy
  selection, which is the premature convergence islands exist to prevent. Set a floor
  before an island is eligible to die.
- **The anchor island is exempt.** Baseline reproduction will be outperformed by design
  once anything works, and it is both a graded requirement and everyone's reference point.
  Retire it deliberately once it confirms 0.6016; do not let extinction take it.
- **Freed budget redistributes to the live islands.** With a small hardcoded set there is
  nothing new to respawn into.

**Migration — mostly sealed, with a rare evidence-gated exception.**

Migration is the island-model mechanism where a trait discovered on one island is copied
to another, so a good discovery is not trapped where it was born. It is also the exact
mechanism that collapses diversity: migrate freely and every island adopts every trick,
the population converges to one point, and the islands stop buying anything.

The design is **isolation by default**, with migration allowed only when a trait proves
itself broadly. Two gates:

- **Evidence bar (the important one).** A trait qualifies only if it produced a > ε
  improvement on **two or more islands independently**. This is what "universally useful"
  has to mean to be mechanizable.
- **Minimum interval.** At least 2 rounds on an island before it may migrate again, so
  the bar cannot be tripped by noise.

Expect this to fire **rarely, possibly never** — with 3–4 islands and ~10 rounds each,
two independent confirmations is a high bar. That is correct for the diversity goal, but
it means inter-island migration cannot be the primary route to combining traits. The
absorber island below is.

**The absorber island — late-activating, not always-on.**

An "accessible" island that freely takes traits from the isolated islands. Worth having,
but activating it at round one causes three problems:

1. **It has no invariant.** Every other island is defined by the one ingredient it must
   keep. The absorber's identity is "whatever is currently winning", so it is not really
   an island — it is the final merge phase running continuously and consuming slots the
   whole way.
2. **It becomes the extinction yardstick.** Holding every validated trait, it will usually
   score highest, and under "stalled AND outperformed" it kills the specialist islands one
   by one. That reintroduces greedy selection through the back door — the exact dynamic
   islands exist to prevent.
3. **Budget.** Four islands plus an absorber is ~10 experiments each, and the absorber's
   experiments are the least novel of the lot (recombinations of already-validated
   traits). Expensive while still below baseline.

The value is nonetheless real: the remaining headroom probably lives in *combinations* —
listwise loss plus sequence features likely beats either alone — and testing that only
once at the very end is fragile, because a broken final merge leaves no budget to debug.

**Resolution — decided:** the absorber spawns as a **fifth island** partway through the
run, seeded with the validated traits, and the specialists keep running alongside it.
Diversity is protected during exploration, combination effects get tested with slack to
debug them, and the final build is a continuation of a live island rather than a cold
start. Three rules attached:

- The absorber is **exempt from extinction**.
- **No other island may be killed by comparison to the absorber** — extinction comparisons
  are between specialist islands only.
- **The absorber inherits dead islands' slots.** It does not get a hardcoded allocation;
  it accumulates the per-round slots released by extinction.

That last rule is load-bearing. At a flat one slot per round the absorber would own only a
fifth of each round and finish with ~3–4 experiments total — *less* runway than a hard mode
switch would give it, which defeats the point of choosing the fifth-island form. Slot
inheritance fixes this without a schedule: more deaths → more absorber share → the
transition from exploration to combination becomes a **drift** driven by evidence rather
than a cliff at a fixed round.

**Spawn trigger.** Not a fixed round. Two-sided:

```
(trait_ledger has >= k traits at single-island or better
     OR specialist deaths >= threshold
     OR round >= R_max)
AND round >= R_min
```

- **Trait count is the primary signal.** It directly measures "there is material worth
  combining", which is what the absorber needs. Island count is only a proxy for
  "exploration is exhausted".
- **Deaths are an accelerator and the slot source**, not the trigger on their own. With
  only 3 specialists the death threshold has nowhere sensible to sit: "below 3" is a single
  death (weak signal), "below 2" leaves one specialist alive (exploration already
  collapsed). Four specialists would give the threshold room if it is used as a primary
  trigger.
- **`R_max` is a hard backstop.** Extinction requires stalled *and* outperformed *and* past
  grace; if every specialist keeps creeping upward, nothing dies and the absorber would
  never spawn, ending the run with separate specialists and no combined candidate at all.
- **`R_min` is a floor.** Two early stalls should not spawn an absorber with nothing
  validated to combine.

**Budget.** 50 scored experiments is tight, and two adjustments buy the absorber real
runway:

- **Anchor as preflight, not a standing island.** Baseline reproduction runs once, not
  once per round. ~2 experiments total instead of ~12.
- **Three specialists rather than four.** Sealed rounds 1–8 at 3/round = 24 experiments.
  The remaining 24 at (3 specialists + 2 absorber) = 5/round ≈ 5 rounds, giving the
  absorber ~10 experiments — enough to recover from a broken merge. At four specialists it
  lands nearer ~6.

**Traits must be first-class objects for any of this to work.**

Journal records currently carry file-level diffs. "Migrate a trait" against a file diff
means copying the whole model, which violates the receiving island's invariant. Traits
have to be named, described objects the master assigns per experiment:

```json
"traits": ["within-user listwise softmax", "4:1 negative sampling"]
```

Without this, both migration and the absorber are unimplementable as specified.

**Traits carry a validation record, not a boolean.** "Validated" as one bit collapses a
+0.001 result and a +0.01 result into the same thing, when the baseline's 5-seed std is
0.0008 and ε is 0.002 — the first is noise, the second is real. Record per trait:

| Field | Why |
|---|---|
| `confirmations` | Count of islands that independently improved with it. Already the migration gate. |
| `deltas` | Effect size per observation, judged against the 0.0008 noise floor. |
| `refutations` | Islands where it was tried and failed. Without this, the same dud gets re-migrated. |
| `status` | See tiers below. |

Tiers: `unvalidated` → `screened` / `screened-negative` (batch evidence only) →
`single-island` (isolated, delta > ε) → `replicated` (≥2 confirmations, ≥1 isolated —
migration-eligible) → `refuted` (isolated failure) → **`context-dependent`** (helped under
one invariant, hurt under another). That last tier is the most interesting artifact the run
produces and would otherwise be discarded — it is direct evidence about *why* a given
island exists.

**Attribution — decided: screen in batches, confirm in isolation.**

An experiment that introduces two traits produces one score, and the delta cannot be split
between them. The rule is **confidence-dependent**:

- **Low-confidence traits are batched.** Several unvalidated candidates go into one
  experiment and the observed delta is spread across them. Cheap survey — three traits per
  batch screens nine candidates in three experiments instead of nine, which is the only way
  to cover ground on a 50-experiment budget.
- **High-confidence traits are isolated.** One new trait against its parent, nothing else
  changed. Higher validity demands a cleaner environment.

**Batch results are prioritisation, not evidence.** Even credit-spreading is not just
noisy, it is wrong in specific ways:

- A batch scoring +0.009 spread as +0.003 each may really be one trait at +0.012 and two at
  −0.0015. Even spread promotes two duds.
- **Interaction effects.** Two traits that only work together — listwise loss plus sequence
  features is the plausible case here — are understated by the spread, then each fails in
  isolation, then both are refuted and the winning *combination* is lost. This is the
  dangerous one: combinations are where the headroom is argued to be, and they are the
  absorber's whole premise.
- **Masking.** One strongly negative trait sinks the batch and a good trait in it reads as
  a failure.

So a batch writes a **batch-level record** and moves its members `unvalidated → screened`,
never to `single-island`. Only isolated experiments write per-trait deltas. The spread
decides who gets the next isolation slot and nothing more. Symmetrically, a failed batch
marks members `screened-negative` (deprioritised), **not** `refuted` — masking makes batch
refutation unsafe.

Consequent tightening of the migration gate: promotion to `replicated` requires **≥2
confirmations of which ≥1 is isolated**. Without that, two dirty batch results could
promote a dud to migration-eligible.

Two things make this cheaper than it sounds:

- **Isolation is relative, not absolute.** "Isolated" means a single-trait diff against the
  parent; the island invariant stays present, being the island's identity rather than a
  confounder. Selective delegation via `active_sub_agents` already does exactly this, so
  isolation runs need no new machinery.
- **ε doubles as the significance threshold.** ε = 0.002 was derived as ≈2.5σ of the
  baseline's 0.0008 seed std. A trait delta below ε is indistinguishable from seed noise,
  and multi-seed replication costs a slot per seed. Only deltas > ε count as evidence — the
  convergence constant and the trait-significance constant are the same number for the same
  reason.

The trait ledger is also the strongest available exhibit for **Innovation (20%)**, which is
judged on reasoning rather than implementation. Replication counts and effect sizes make a
better case than a list of hypotheses does.

**Journal support.** Cross-island parentage already works — `parent_experiment_id` is
unconstrained. Missing: an `island` field on each record, the island's invariant, and the
`traits` list above, so lineage, migration and extinction decisions are auditable after
the fact. Deliverable 3 wants per-iteration lineage regardless.

### Open design questions

- **What is the "threshold"** that triggers the final-hypothesis phase — a round count, a
  score plateau across all islands, or a fixed reserve of the 50? It must fire early
  enough to leave room for the final build (conflict 2).
- **Grace-period and stall-window lengths.** Both are round counts and both interact with
  the 50-experiment budget; at 4 islands there are only 12 rounds to spend.
- **Absorber trigger constants** — `k` (validated traits), death threshold, `R_min`,
  `R_max`. All fall out once the island count is fixed. `R_max` sketched at round 8 of ~12.
- **Three specialists or four.** Three gives the absorber more runway per the budget note
  above; four gives the death-threshold trigger somewhere to sit.
- **Batch size for screening.** Three traits per batch is the working assumption above;
  larger batches survey more but worsen masking and interaction confounding.

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
| Guardrails | `sandbox.py` | Path containment, per-role allowlist, literal SEARCH/REPLACE edits |
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
