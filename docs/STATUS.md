# Status — where the system actually is

Last updated: 2026-08-31. Update this file whenever a run finishes.

## Code layout

| Path | What it is |
|---|---|
| `PROBLEM.md` | Official problem statement, verbatim. Read-only reference. |
| `docs/BENCHMARK.md` | Distilled task, scoring, constraints, headroom, judging. |
| `docs/DESIGN.md` | Agent architecture and the decisions behind it. |
| `docs/STATUS.md` | This file. |
| `src/vibes/` | **Active codebase.** Copy of `src/ernest/` — all new work goes here. |
| `src/ernest/` | Ernest's original; frozen reference. Byte-identical to `vibes` as of the copy. |
| `src/.tw/` | tw's design (`agents.md`) + ~530 lines of prototype. Idea source, not a dependency. |
| `src/kuairand-starter-kit/` | Organizer starter kit + `KuaiRand-Pure/` data (already downloaded). |

## Run history

Both runs are on **validation** (124,909 rows). Baseline validation primary = **0.6016**.

| Run | Scored | Abandoned | Best primary | vs baseline |
|---|---|---|---|---|
| run_1 | 1 | 2 | 0.5745 | −0.0271 |
| run_2 | 3 | 16 | 0.5885 | −0.0131 |

**Not beating baseline yet.** Two failures explain nearly all of it.

---

## Failure 1 — patch application eats the iteration budget

Every single abandonment in run_2 was a patch failure. Zero were caused by bad science
or bad ML:

| Count | Stage |
|---|---|
| 8 | `initial_patch:model_designer` — failed to provide an applicable patch |
| 5 | `initial_patch:feature_engineer` — same |
| 3 | `trainer` (1 initial, 2 debug) — same |

Representative reason, verbatim:

```
RuntimeError: model_designer failed to provide an applicable patch after three
responses: ValueError: patch context does not match reference file
```

The generated *code* is fine. Inspecting `patch_history.json` on
`run_2/abandoned/attempt_04dcc668a01c`, the model wrote a clean, plausible
within-user listwise softmax with correct gradient bookkeeping — and it was thrown
away because the unified-diff hunk context didn't line up. Three repair attempts,
all rejected the same way.

**Consequence:** 12 of 18 hypotheses were never tested at all. The ones that died
include BPR, listwise softmax, sequential user features, multi-task auxiliary heads,
censored watch-time — i.e. **exactly the organizers' top-4 unexplored directions**
(see `docs/BENCHMARK.md` §6). The agent proposed the right things. The plumbing threw
them away.

This is the single highest-value fix in the repo. See `docs/DESIGN.md` §"Patch format".

**Fixed 2026-08-31 (Phase 0 #3).** The agent-facing edit format is now literal SEARCH/REPLACE
blocks (`sandbox.py:apply_search_replace`). No hunk headers, no line arithmetic. A SEARCH that
matches zero or more than one place is rejected as a contract failure and routed back through the
existing 3-response repair loop with the reason; it is never applied to the first hit. Path
containment and the per-role allowlist are unchanged. The journal still stores a unified diff,
computed against the parent in `overseer.py`.

## Failure 2 — the seed model has a dead interaction term

`src/vibes/agentic_recsys/seed/model.py` zero-initialises every parameter. That is
harmless for the purely additive seed. It is fatal the moment a descendant adds a
multiplicative interaction, which is the first thing any sensible agent does:

```python
self.user_embeddings = np.zeros((user_dimension, 16), dtype=np.float32)
self.item_embeddings = np.zeros((item_dimension, 16), dtype=np.float32)
```

Gradient w.r.t. `user_embeddings` is proportional to `item_embeddings` and vice versa.
Both start at zero, so both stay at zero forever. The interaction term is inert.

**Proof it bit:** run_2 experiment_1 and experiment_2 tested two different hypotheses
(a low-rank interaction term, and a compact FM interaction term) and returned
*bit-identical* scores — `0.5885421318404991` both times. Neither interaction ever
learned anything. Both "scored" experiments are the same degenerate additive model.
Real progress across run_2 is **zero**.

The agent did eventually diagnose this itself — generation 2 proposed "break the
zero-gradient symmetry by initializing only one side" — and then lost both attempts to
Failure 1.

**Fixed 2026-08-31 (Phase 0 #2).** The seed initialises from a small seeded random normal
(`init_scale`, now a config key), and `train.py` verifies after training that every block named
by `Model.parameter_blocks()` actually changed. A dead-gradient model now raises, naming the
inert blocks and `model.py`, which routes the repair to the model_designer.

Verified against a reconstruction of run_2's failure: a zero-initialised multiplicative factor
block trains to completion and is caught (`dead gradient in model.py: parameter block(s)
['factors'] did not change during training`); the same architecture with a non-zero init trains
and passes. The seed itself now scores **primary 0.5885203** on validation (GAUC 0.6498,
nDCG@5 0.5272) — which is run_2's 0.5885421 to four decimals, confirming that both of run_2's
"scored" experiments were the untouched additive seed.

---

## Consequences for judging

Mapping the above onto `docs/BENCHMARK.md` §7:

- **Technical Execution (35%)** — currently below baseline, so the primary delta is
  negative. Both root causes are plumbing, not research.
- **Feasibility (15%)** — gated on beating the baseline. Currently not scored at all.
  16 abandoned attempts also burn tokens and wall-clock for nothing.
- **Innovation (20%)** — judged on *reasoning*, not implementation. The hypothesis
  quality in the journal is genuinely good and already targets the organizers'
  priority list. This is our strongest axis right now, and the journal already
  captures it.
- **Impact/Autonomy (20%)** — no manual interventions in either run. Also strong.
  Keep it that way and keep counting.

## Open questions

1. **Deliverable 3 wants per-iteration hypothesis + diff + metrics + error/recovery
   events.** Our journal already carries all four fields. Confirm the export format
   before submission rather than after. Note the logged diff can be *computed* — it need
   not be the format the agent emitted (see `docs/DESIGN.md`, patch format).
2. **Bonus benchmarks** (KuaiRand-1k / 27k) — not attempted, worth nothing lost.
   Only consider after Pure clears the baseline.

## Next actions

Design is settled at the mechanism level (`docs/DESIGN.md`). What remains open is numeric
constants — make them config keys with defaults and pick them once a round's real
wall-clock and token cost is known, not in the abstract.

Test suite is green: `python -m unittest discover -s tests` → 23 tests, OK. That is the
safety net for the patch-format refactor.

### Phase 0 — plumbing, before any island scaffolding

**All four items landed 2026-08-31.** Test suite went 23 → 53, green. No manual interventions
during a run (nothing has been run yet); these are design-time changes.

1. **DONE — `journal.py` → running best.** Smallest change, and it is what truncated run_2 to
   3 of 50. An island run explores by design, so the raw-score rule would fire constantly.
   See `docs/DESIGN.md` "Four conflicts" #1.
   Landed as `Journal.running_best()`, the cumulative maximum of validation primary over
   scored experiments; `converged()` measures the window on that series, so it is monotone
   non-decreasing and a dud can only fail to advance it. A genuine plateau still converges.
   `Journal.best_record()` names the validation-best checkpoint — the one that gets
   submitted — and `Overseer.run()` now reports `best_primary`/`best_experiment_id`
   alongside the existing last-scored fields.
2. **DONE — Seed init + parameter-movement smoke check.** Otherwise a dead-gradient experiment
   silently reports its parent's score — which would poison the trait ledger, since a
   trait that changes nothing reads as neutral rather than broken.
3. **DONE — Patch format → search/replace blocks.** The big one: 100% of run_2's abandonments.
   Unblocks the 12 hypotheses never tested, including every ranking-loss idea the island
   design is built around. Do it third, with 1 and 2 already green.

4. **DONE — Instrument token usage and model name in `llm_events.jsonl`.** run_2's 122 events
   recorded only `context, payload, response, role, sequence, status, system, timestamp`.
   Deliverable 4 requires total input + output tokens, and Feasibility (15%) is scored on it.
   Cannot be reconstructed after the fact.
   Every event now also carries `model` (the *served* model id, which may be a dated
   snapshot of the requested one), `usage`, and `duration_seconds`. Responses-API and
   Chat-Completions usage shapes are normalised onto one set of names
   (`input_tokens`, `output_tokens`, `total_tokens`, `cached_input_tokens`,
   `reasoning_tokens`) by `llm.normalize_usage`. Errored calls are logged and counted but
   never carry the previous call's usage. Run totals come from
   `AuditedLLMClient.usage_report()` (in the `Overseer.run()` summary as `token_usage`) and
   `ernest status <run_dir>` re-derives the same totals from the event log.

**Runtime prerequisite.** The agent needs `OPENAI_API_KEY` (or `--api-key`);
`OpenAICompatibleClient` in `llm.py` raises without one. Alternatives: `--base-url` plus `--api-mode chat` / `--no-json-mode`
for any chat-completions-compatible provider, or `--llm-command` for a local adapter with
no key at all. Neither env var nor `.env` is present on the Windows checkout. `.gitignore`
covers `.env` / `*.env` — keep it that way, deliverable 2 requires the repo be public.
Prior runs were executed from `/mnt/d/tehpengagent/src/ernest` under WSL, i.e. a different
machine and checkout.

**Phase 0 verification — still outstanding, blocked on an API key.** One run confirming a ranking-loss hypothesis actually executes
end to end, and that the FM baseline is reproducible at validation 0.6016. Nobody has
checked the latter yet, and it is both task requirement 1 and the anchor island's entire
job — better to find out now than during the first island run.

### Phase 1 — island layer, in dependency order

1. Journal schema: `island`, `traits`, trait validation record.
2. Declarative island config (name, invariant, seed scaffold, prompt fragment, literature).
3. Master: keep/kill, extinction with grace period.
4. Absorber: spawn trigger, slot inheritance.
5. Research Agent with per-island RAG namespace.
6. Data Scientist agent (EDA stage).

Ordering matters most for Phase 0 #3: with patch application broken, an island run
produces a well-structured architecture that never completes an experiment, and the
wall-clock is spent discovering that.

## Settled

- **Abandoned attempts do not count against the 50-iteration cap** (decided 2026-08-30).
  Only scored experiments consume a slot; abandoned attempts are bounded by the retry
  ceiling and per-experiment wall-clock instead. run_2 therefore used 3 of 50, not 19.
  State this interpretation explicitly in the final report.
