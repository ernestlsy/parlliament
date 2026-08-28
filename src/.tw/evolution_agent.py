"""
Evolution Judge -- decides which experiment to build and run next by
treating the experiment log as an evolving population of model configs,
not just "ask an LLM for the next JSON blob":

1. Selection: pick a parent from the population of past experiments.
2. Exploit vs explore: if recent iterations have stalled against the
   running best, force a branch to a different architecture (explore);
   otherwise mutate the current best (exploit). This is the exploit/explore
   call agents.md assigns to the Evolution Judge.
3. Mutation: the LLM proposes one concrete child ExperimentSpec (a small
   hyperparameter perturbation when exploiting, a architecture change when
   exploring), grounded in the parent's own numbers and the Research
   Agent's findings.

Falls back to a deterministic mutation (no LLM) if the model can't produce
valid JSON twice in a row, so the loop always keeps building and improving
something instead of stalling on a flaky free-tier LLM call.
"""
import json
import re
import time

from pydantic import BaseModel, Field

STALL_EXPLORE_AFTER = 2  # consecutive non-improving iterations before forcing exploration

DOMAIN_BRIEFING = """
Task: rank each user's logged video impressions by predicted relevance
(label = long_view). Metric = primary = mean(GAUC, nDCG@5), scored on the
held-out valid split. Oracle ceiling is 0.8645 primary on test, so treat
headroom above the FM baseline as ~0.27, not ~0.40.

Known facts -- already tested, do not re-propose these:
- FM official baseline: k=16, lr=0.001, epochs<=40 -> test primary 0.5946.
- Adding more static features (13 fields instead of 5) did NOT help
  (0.5940 vs 0.5950 -- within noise, slightly worse).
- Bigger embeddings (k=8/16/32) did NOT help (0.5895/0.5902/0.5887 -- flat).

You may propose ANY architecture name -- it is not restricted to a fixed list.
If it isn't actually implemented, the run will fail cleanly and you'll see
that in the history below, so you can course-correct next time. Three are
implemented and runnable for real right now:
- "fm": pointwise logistic Factorization Machine (the baseline). Cheap,
  ~1s/epoch.
- "bpr": pairwise BPR-ranking RNN over the same 5 fields (user_id, video_id,
  author_id, tab, dur_bucket). Targets the hypothesis that a ranking-aligned
  pairwise loss should beat pointwise logloss, since GAUC/nDCG are themselves
  ranking metrics. ~7s/epoch in Python, so prefer epochs <= 15 unless a
  longer run is clearly justified by the trend.
- "deepfm": DeepFM -- the same shared field embeddings feed both the FM
  2nd-order interaction (as in "fm") AND a deep MLP over the flattened
  embeddings, with both outputs summed before the sigmoid. This is the
  README's "change model" direction: the deep component can capture
  higher-order, non-linear feature interactions that plain FM cannot. ~5s/epoch.
"""


class ExperimentSpec(BaseModel):
    architecture: str = Field(description="Any architecture name; only ones the Compute "
                                           "Manager implements can actually be run")
    hypothesis: str = Field(description="What are we testing and why")
    parent_id: int
    k: int = Field(ge=4, le=64)
    lr: float = Field(gt=0, le=0.05)
    epochs: int = Field(ge=1, le=25)
    seed: int = 0


def clean_json(raw: str) -> str:
    return re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip(), flags=re.IGNORECASE).strip()


def _ok(history) -> list:
    return [e for e in history.history if e.get("status") == "ok"]


def _decide_mode(history) -> str:
    ok = _ok(history)
    if len(ok) <= STALL_EXPLORE_AFTER:
        return "exploit"
    best_so_far, stall = ok[0]["valid"]["primary"], 0
    for e in ok[1:]:
        p = e["valid"]["primary"]
        stall = stall + 1 if p <= best_so_far else 0
        best_so_far = max(best_so_far, p)
    return "explore" if stall >= STALL_EXPLORE_AFTER else "exploit"


def _select_parent(history, mode):
    """exploit -> current best (refine a winner). explore -> the best entry
    among architectures OTHER than the current best's, forcing a lineage
    branch instead of re-mutating the same winner; falls back to the overall
    best if only one architecture has ever run."""
    ok = _ok(history)
    if not ok:
        return None
    best = max(ok, key=lambda e: e["valid"]["primary"])
    if mode == "exploit":
        return best
    others = [e for e in ok if e["architecture"] != best["architecture"]]
    return max(others, key=lambda e: e["valid"]["primary"]) if others else best


def propose_experiment(client, model, history, next_id: int, research_text: str,
                        available_architectures: list) -> ExperimentSpec:
    mode = _decide_mode(history)
    parent = _select_parent(history, mode)
    parent_id = parent["id"] if parent else 0
    print(f"[Evolution Judge] mode={mode} parent={parent_id}"
          + (f" ({parent['architecture']}, valid_primary={parent['valid']['primary']:.4f})" if parent else " (no history yet)"))

    if parent is None:
        directive = "No successful experiment yet -- propose a sensible first real experiment."
    elif mode == "exploit":
        directive = (
            f"EXPLOIT: recent iterations are still improving. Build on experiment id={parent_id} "
            f"(architecture={parent['architecture']}, config={parent['config']}, "
            f"valid_primary={parent['valid']['primary']:.4f}). Keep its architecture, and make a "
            f"small, deliberate change to one or two hyperparameters (e.g. roughly +/-30% on k or "
            f"lr, or a handful more/fewer epochs) that your hypothesis predicts will help further."
        )
    else:
        directive = (
            f"EXPLORE: recent iterations have stalled at or below the running best. Do NOT just "
            f"re-mutate the current best again -- branch off a different lineage. Propose a "
            f"different architecture than '{parent['architecture']}' "
            f"(parent for lineage purposes: id={parent_id}, config={parent['config']})."
        )

    schema_hint = (
        '{"architecture": "string, any name", "hypothesis": "string", '
        f'"parent_id": {parent_id}, "k": int (4-64), "lr": float (0-0.05), '
        '"epochs": int (1-25), "seed": int}'
    )
    prompt = (
        f"{DOMAIN_BRIEFING}\n{research_text}\n\n"
        f"Evolution Judge directive -- {mode.upper()} MODE:\n{directive}\n\n"
        f"Use the research findings above as supporting evidence in your hypothesis where relevant.\n\n"
        f"Full experiment history so far:\n{history.summary()}\n\n"
        f"Respond with ONLY a raw JSON object matching this schema, no markdown "
        f"fences, no commentary:\n{schema_hint}"
    )
    messages = [
        {"role": "system", "content": "You are the Evolution Judge in an evolutionary ML experiment "
                                       "loop -- a precise automated JSON API, not a chatbot."},
        {"role": "user", "content": prompt},
    ]
    for attempt in range(3):
        try:
            response = client.chat.completions.create(model=model, messages=messages)
            raw = response.choices[0].message.content
            data = json.loads(clean_json(raw))
            data.setdefault("parent_id", parent_id)
            return ExperimentSpec(**data)
        except Exception as e:
            print(f"  [Evolution Judge] attempt {attempt + 1} failed: {e}")
            if "429" in str(e) or "rate" in str(e).lower():
                time.sleep(6)  # OpenRouter free-tier shared pool is often briefly rate-limited
            else:
                messages.append({"role": "user",
                                  "content": f"That failed to parse ({e}). Return ONLY valid raw JSON matching the schema."})

    # LLM did not produce a usable spec -- fall back to a deterministic version
    # of the same directive so the loop still evolves instead of stalling.
    print(f"  [Evolution Judge] LLM proposal failed, falling back to a deterministic {mode} move.")
    base_cfg = parent["config"] if parent else {"k": 16, "lr": 0.001, "epochs": 15, "seed": 0}
    if mode == "exploit" and parent:
        next_arch = parent["architecture"]
        k = max(4, min(64, round(base_cfg.get("k", 16) * 1.3)))
        lr = base_cfg.get("lr", 0.001)
    else:
        prev = parent["architecture"] if parent else None
        remaining = [a for a in available_architectures if a != prev] or available_architectures
        next_arch = remaining[next_id % len(remaining)]
        k, lr = 16, 0.001
    return ExperimentSpec(
        architecture=next_arch,
        hypothesis=f"Fallback ({mode}, LLM proposal failed): "
                    f"{'mutate' if mode == 'exploit' else 'branch to'} {next_arch} with "
                    f"{'perturbed' if mode == 'exploit' else 'baseline'} hyperparameters.",
        parent_id=parent_id,
        k=k, lr=lr,
        epochs=min(base_cfg.get("epochs", 15), 15),
        seed=next_id,
    )
