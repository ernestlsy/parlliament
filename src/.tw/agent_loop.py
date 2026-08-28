"""
Simple autonomous agent workflow for the KuaiRand-Pure ranking task.

Orchestrator -> Research Agent (live arXiv lookup on a rotating topic)
             -> Evolution Judge (selects a parent from the experiment
                population, decides exploit-vs-explore, has the LLM mutate
                or branch it into the next experiment -- see evolution_agent.py)
             -> Compute Manager (actually builds/trains/evaluates the
                proposed model on real data)
             -> Persistent Judge (logs every experiment to disk)
repeats until --iterations is reached or the official convergence rule fires
(baseline_scores.json: 3 consecutive iterations improving valid primary by
<= 0.002). This is a working subset of the 8-agent design in agents.md --
Data Scientist / Feature Engineer are left as future extensions. The
Evolution Judge can propose ANY architecture name; only the ones in
IMPLEMENTED (currently fm, bpr, deepfm) can actually be run -- anything
else fails cleanly and the LLM sees why in the next iteration's history.
"""
import argparse
import json
import os
import re
import sys
import time

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

# LLM output can contain arbitrary Unicode (smart quotes, en/em dashes, ...)
# that Windows' default cp1252 console can't print -- widen stdout so a
# print() never crashes the loop.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
KIT_DIR = os.path.abspath(os.path.join(HERE, "..", "kuairand-starter-kit"))
sys.path.insert(0, KIT_DIR)

from data import load, encode                  # noqa: E402
from baseline import run_fm                    # noqa: E402
from tw_model import run_bpr, run_deepfm       # noqa: E402

sys.path.insert(0, HERE)
from research_agent import get_findings, format_findings          # noqa: E402
from evolution_agent import propose_experiment, ExperimentSpec    # noqa: E402

load_dotenv(os.path.join(HERE, ".env"))

DATA_DIR = os.path.join(KIT_DIR, "KuaiRand-Pure", "data")
LOG_PATH = os.path.join(HERE, "experiment_log.json")
BASELINE_SCORES_PATH = os.path.join(KIT_DIR, "baseline_scores.json")

CONVERGENCE_EPS = 0.002
CONVERGENCE_N = 10

# Compute Manager dispatch: what the Evolution Judge can actually run for
# real, keyed by a normalized (lowercase, alnum-only) architecture name.
IMPLEMENTED = {
    "fm": run_fm,
    "bpr": run_bpr,
    "deepfm": run_deepfm,
}


def normalize_arch(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)
MODEL = os.getenv("OPENROUTER_MODEL", "minimax/minimax-m2.7:free")


def _json_default(o):
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.integer):
        return int(o)
    raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")


# ==========================================
# Persistent Judge -- remembers every experiment across runs
# ==========================================
class ExperimentHistory:
    def __init__(self, filepath=LOG_PATH):
        self.filepath = filepath
        self.history = []
        self.load()

    def load(self):
        if os.path.exists(self.filepath):
            with open(self.filepath, encoding="utf-8") as f:
                try:
                    self.history = json.load(f)
                except json.JSONDecodeError:
                    self.history = []

    def log(self, entry: dict):
        self.history.append(entry)
        with open(self.filepath, "w", encoding="utf-8") as f:
            json.dump(self.history, f, indent=2, default=_json_default)

    def best(self):
        ok = [e for e in self.history if e.get("status") == "ok"]
        return max(ok, key=lambda e: e["valid"]["primary"]) if ok else None

    def summary(self, n=8):
        lines = []
        for e in self.history[-n:]:
            if e.get("status") == "ok":
                lines.append(
                    f"id={e['id']} parent={e['parent']} arch={e['architecture']} "
                    f"cfg={e['config']} valid_primary={e['valid']['primary']:.4f} "
                    f"| {e['hypothesis']}"
                )
            else:
                lines.append(f"id={e['id']} arch={e.get('architecture')} FAILED: {e.get('error')}")
        return "\n".join(lines)


# ==========================================
# Compute Manager -- actually trains and evaluates on real data
# ==========================================
def run_experiment(spec: ExperimentSpec, splits, enc, dim) -> dict:
    fn = IMPLEMENTED.get(normalize_arch(spec.architecture))
    if fn is None:
        raise ValueError(f"architecture '{spec.architecture}' is not implemented "
                          f"(available: {', '.join(IMPLEMENTED)})")
    return fn(splits, k=spec.k, lr=spec.lr, epochs=spec.epochs, seed=spec.seed,
              enc=enc, dim=dim, verbose=False)


# ==========================================
# Orchestrator
# ==========================================
def seed_baseline(history: ExperimentHistory):
    if history.history:
        return
    with open(BASELINE_SCORES_PATH, encoding="utf-8") as f:
        scores = json.load(f)["scores"]["fm_official"]
    history.log({
        "id": 0, "parent": None, "status": "ok",
        "architecture": "fm",
        "hypothesis": "Official FM baseline (read from baseline_scores.json, not re-run).",
        "config": scores["config"], "valid": scores["valid"], "test": scores["test"],
        "runtime_sec": 0.0,
    })


def converged(history: ExperimentHistory) -> bool:
    ok = [e for e in history.history if e.get("status") == "ok"]
    if len(ok) < CONVERGENCE_N + 1:
        return False
    best_so_far = ok[0]["valid"]["primary"]
    stall = 0
    for e in ok[1:]:
        p = e["valid"]["primary"]
        stall = stall + 1 if p - best_so_far <= CONVERGENCE_EPS else 0
        best_so_far = max(best_so_far, p)
    return stall >= CONVERGENCE_N


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iterations", type=int, default=5)
    ap.add_argument("--data_dir", default=DATA_DIR)
    args = ap.parse_args()

    print(f"[Orchestrator] Loading KuaiRand-Pure from {args.data_dir} ...")
    splits = load(args.data_dir)
    print({k: len(v) for k, v in splits.items()})
    print("[Orchestrator] Encoding features (shared across all experiments this run) ...")
    enc, dim = encode(splits)

    history = ExperimentHistory()
    seed_baseline(history)
    print(f"[Persistent Judge] {len(history.history)} experiment(s) loaded from {LOG_PATH}")

    next_id = max(e["id"] for e in history.history) + 1

    for i in range(args.iterations):
        if converged(history):
            print(f"\n[Orchestrator] Converged: last {CONVERGENCE_N} iterations improved valid "
                  f"primary by <= {CONVERGENCE_EPS}. Stopping.")
            break

        print(f"\n--- ORCHESTRATOR: ITERATION {i + 1}/{args.iterations} (experiment id {next_id}) ---")
        findings = get_findings(next_id)
        research_text = format_findings(findings)
        print(research_text)

        spec = propose_experiment(client, MODEL, history, next_id, research_text, list(IMPLEMENTED))
        print(f"[Evolution Judge] arch={spec.architecture} k={spec.k} lr={spec.lr} "
              f"epochs={spec.epochs} seed={spec.seed}")
        print(f"  hypothesis: {spec.hypothesis}")

        research_meta = {
            "research_topic": findings["topic"],
            "research_papers": [p["title"] for p in findings["papers"]],
        }

        t0 = time.time()
        try:
            result = run_experiment(spec, splits, enc, dim)
            entry = {
                "id": next_id, "parent": spec.parent_id, "status": "ok",
                "architecture": spec.architecture, "hypothesis": spec.hypothesis,
                "config": {"k": spec.k, "lr": spec.lr, "epochs": spec.epochs, "seed": spec.seed},
                "valid": result["valid"], "test": result["test"],
                "runtime_sec": round(time.time() - t0, 1),
                **research_meta,
            }
            print(f"[Compute Manager] valid primary={result['valid']['primary']:.4f} "
                  f"(GAUC={result['valid']['GAUC']:.4f} nDCG@5={result['valid']['nDCG@5']:.4f}) "
                  f"in {entry['runtime_sec']}s")
        except Exception as e:
            entry = {
                "id": next_id, "parent": spec.parent_id, "status": "failed",
                "architecture": spec.architecture, "hypothesis": spec.hypothesis,
                "config": {"k": spec.k, "lr": spec.lr, "epochs": spec.epochs, "seed": spec.seed},
                "error": str(e), "runtime_sec": round(time.time() - t0, 1),
                **research_meta,
            }
            print(f"[Compute Manager] experiment failed: {e}")

        history.log(entry)
        next_id += 1

    best = history.best()
    print("\n[Orchestrator] Loop finished.")
    if best:
        print(f"Best so far: id={best['id']} arch={best['architecture']} "
              f"valid primary={best['valid']['primary']:.4f} "
              f"test primary={best['test']['primary']:.4f}")
        print(f"  hypothesis: {best['hypothesis']}")
    print(f"Full history: {LOG_PATH}")


if __name__ == "__main__":
    main()
