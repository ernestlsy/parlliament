import json
import tempfile
import unittest
from pathlib import Path

from agentic_recsys.journal import Journal
from agentic_recsys.llm import AuditedLLMClient, LLMClient
from agentic_recsys.schemas import JournalRecord
from agentic_recsys.submission import export_submission_bundle


class UsageClient(LLMClient):
    def __init__(self):
        self.last_usage = None

    def complete_json(self, *, role, system, payload):
        self.last_usage = {
            "input_tokens": 11,
            "output_tokens": 7,
            "total_tokens": 18,
        }
        return {"ok": True}


class SubmissionTests(unittest.TestCase):
    def test_audited_client_persists_provider_usage(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "llm_events.jsonl"
            AuditedLLMClient(UsageClient(), path).complete_json(
                role="judge", system="return JSON", payload={}
            )
            event = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(event["usage"]["total_tokens"], 18)

    def test_bundle_contains_logs_results_resources_and_manual_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "run_1"
            sandbox = run_dir / "experiment_1"
            sandbox.mkdir(parents=True)
            (sandbox / "attempt_summary.json").write_text(json.dumps({
                "elapsed_seconds": 4.5,
                "failure_reports": [{
                    "kind": "semantic_logic",
                    "message": "shape mismatch",
                    "responsible_agents": ["model_designer"],
                    "attempt": 1,
                }],
            }), encoding="utf-8")
            Journal(run_dir / "journal.jsonl").append(JournalRecord(
                attempt_id="attempt-a",
                experiment_id=1,
                generation=1,
                parent_experiment_id=0,
                hypothesis_text="Try a ranking-aware objective because GAUC is weak.",
                hypothesis_scores={"interestingness": 8, "novelty": 7, "feasibility": 6},
                mode="draft",
                code_diff={"model.py": "--- model.py\n+++ model.py\n@@\n-old\n+new\n"},
                config_diff="",
                active_sub_agents=["model_designer"],
                metrics={
                    "GAUC": 0.68,
                    "nDCG@5": 0.55,
                    "primary": 0.615,
                    "segment_diagnostics": {"status": "available", "segments": {}},
                },
                status="scored",
                failure_reason=None,
                failure_stage=None,
                consultant_rounds=1,
                sandbox=str(sandbox),
                created_at="2026-01-01T00:00:00+00:00",
            ))
            (run_dir / "system_config.json").write_text(
                json.dumps({"data_dir": str(Path(directory) / "data")}), encoding="utf-8"
            )
            (run_dir / "run_timing.json").write_text(
                json.dumps({"total_wall_clock_seconds": 123.5}), encoding="utf-8"
            )
            (run_dir / "llm_events.jsonl").write_text(json.dumps({
                "status": "success",
                "usage": {"input_tokens": 100, "output_tokens": 20, "total_tokens": 120},
            }) + "\n", encoding="utf-8")

            manifest = export_submission_bundle(run_dir)
            self.assertEqual(manifest["status"], "incomplete")
            self.assertIn("predictions_test.npz", manifest["submission_error"])
            log = (run_dir / "submission" / "iteration_log.md").read_text(encoding="utf-8")
            self.assertIn("Manual interventions: **0 (none)**", log)
            self.assertIn("shape mismatch", log)
            iteration_log = json.loads(
                (run_dir / "submission" / "iteration_log.json").read_text(encoding="utf-8")
            )
            self.assertNotIn("segment_diagnostics", iteration_log["attempts"][0]["metrics"])
            results = json.loads(
                (run_dir / "submission" / "results.json").read_text(encoding="utf-8")
            )
            self.assertAlmostEqual(results["result"]["absolute_delta_GAUC"], 0.0126)
            self.assertEqual(results["resources"]["token_consumption"]["total_tokens"], 120)
            self.assertEqual(results["resources"]["agent_wall_clock_seconds"], 123.5)
            self.assertEqual(results["resources"]["manual_interventions"], 0)


if __name__ == "__main__":
    unittest.main()
