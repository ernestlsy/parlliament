# Status — where the system actually is

Last updated: 2026-08-30. Update this file whenever a run finishes.

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
| run_2 | 3 | 15 | 0.5885 | −0.0131 |

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

**Fix:** small random init (or one-sided random init) in the seed, and a smoke check
that a parameter block actually moves during training.

---

## Consequences for judging

Mapping the above onto `docs/BENCHMARK.md` §7:

- **Technical Execution (35%)** — currently below baseline, so the primary delta is
  negative. Both root causes are plumbing, not research.
- **Feasibility (15%)** — gated on beating the baseline. Currently not scored at all.
  15 abandoned attempts also burn tokens and wall-clock for nothing.
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

## Next actions, in order

1. Replace unified diffs with search/replace blocks (`docs/DESIGN.md`). Unblocks the 12
   untested hypotheses.
2. Fix seed initialisation + add a "did any parameter block actually move?" smoke check,
   so a dead-gradient experiment fails loudly instead of silently echoing its parent.
3. Re-run and confirm the ranking-loss hypotheses (BPR, within-user listwise softmax)
   now survive to execution.
4. Only then: add the Research Agent and Data Scientist Agent.

## Settled

- **Abandoned attempts do not count against the 50-iteration cap** (decided 2026-08-30).
  Only scored experiments consume a slot; abandoned attempts are bounded by the retry
  ceiling and per-experiment wall-clock instead. run_2 therefore used 3 of 50, not 18.
  State this interpretation explicitly in the final report.
