import csv
import json
import tempfile
import unittest
from pathlib import Path

from agentic_recsys.config import SystemConfig
from agentic_recsys.llm import ScriptedLLMClient
from agentic_recsys.overseer import Overseer


def write_csv(path, header, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


class EndToEndTests(unittest.TestCase):
    def test_one_counted_experiment(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "data"
            data.mkdir()
            write_csv(
                data / "video_features_basic_pure.csv",
                ["video_id", "author_id"],
                [["v1", "a1"], ["v2", "a2"]],
            )
            header = ["date", "user_id", "video_id", "tab", "duration_ms", "long_view"]
            write_csv(
                data / "log_standard_4_08_to_4_21_pure.csv",
                header,
                [
                    [20220408, "u1", "v1", "1", 1000, 1],
                    [20220408, "u1", "v2", "1", 2000, 0],
                    [20220409, "u2", "v1", "1", 1000, 0],
                    [20220409, "u2", "v2", "1", 2000, 1],
                ],
            )
            write_csv(
                data / "log_standard_4_22_to_5_08_pure.csv",
                header,
                [
                    [20220422, "u1", "v1", "1", 1000, 1],
                    [20220422, "u1", "v2", "1", 2000, 0],
                    [20220422, "u2", "v1", "1", 1000, 0],
                    [20220422, "u2", "v2", "1", 2000, 1],
                    [20220429, "u1", "v1", "1", 1000, 1],
                ],
            )
            contract = {
                "data_output": {"splits": ["train", "valid", "test"]},
                "config_keys": [
                    "seed", "learning_rate", "l2", "batch_size",
                    "max_epochs", "patience", "split",
                ],
                "model_input": {"features": "int32[N,F]"},
                "prediction_artifact": {
                    "path": "predictions_valid.npz",
                    "arrays": ["row_ids", "scores"],
                },
                "train_command": [
                    "{python}", "train.py", "--config", "config.json",
                    "--data-dir", "{data_dir}", "--output", "{output}",
                ],
                "contract_command": [
                    "{python}", "train.py", "--config", "config.json",
                    "--data-dir", "{data_dir}", "--output", "{output}",
                    "--contract-check",
                ],
            }
            llm = ScriptedLLMClient([
                {"priorities": [{
                    "evidence_ids": ["screen:item_metadata"],
                    "finding": "item metadata is stable",
                    "recommended_action": "use it",
                }], "avoid": [], "metric_diagnosis": "smoke test"},
                {"candidates": [{
                    "candidate_id": "c1",
                    "text": "Document the neutral seed without changing its behavior",
                    "parent_experiment_id": 0,
                    "scores": {"interestingness": 2, "novelty": 2, "feasibility": 10},
                    "rationale": "Pipeline smoke test",
                    "evidence_ids": ["screen:item_metadata"],
                    "exact_ablation": "change the model docstring",
                    "expected_effect": {"GAUC": "neutral", "nDCG@5": "neutral"},
                    "expected_primary_gain": 0.0,
                    "confidence": 10,
                    "leakage_risk": "low",
                    "runtime_risk": "low",
                    "active_components": ["model_designer"],
                }, {
                    "candidate_id": "c2",
                    "text": "Change only the learning rate",
                    "parent_experiment_id": 0,
                    "scores": {"interestingness": 3, "novelty": 3, "feasibility": 9},
                    "rationale": "Distinct smoke candidate",
                    "evidence_ids": ["screen:item_metadata"],
                    "exact_ablation": "change only learning rate",
                    "expected_effect": {"GAUC": "unknown", "nDCG@5": "unknown"},
                    "expected_primary_gain": -0.001,
                    "confidence": 5,
                    "leakage_risk": "low",
                    "runtime_risk": "low",
                    "active_components": ["trainer"],
                }]},
                {"ranking": [
                    {"candidate_id": "c1", "rank": 1, "utility_score": 9, "rationale": "safe"},
                    {"candidate_id": "c2", "rank": 2, "utility_score": 3, "rationale": "weaker"},
                ]},
                {"winner_candidate_id": "c1", "selection_rationale": "highest confidence"},
                {"active_agents": ["model_designer"], "reasoning": "model-only", "contract": contract},
                {"patches": {"model.py": (
                    "--- model.py\n+++ model.py\n@@ -1,1 +1,1 @@\n"
                    "-\"\"\"Fresh-start additive ID model with no interaction or baseline-derived architecture.\"\"\"\n"
                    "+\"\"\"Fresh-start additive ID model with no interaction or inherited architecture.\"\"\"\n"
                    "BROKEN PATCH TRAILER\n"
                )}},
                {"patches": {"model.py": (
                    "--- model.py\n+++ model.py\n@@ -1,1 +1,1 @@\n"
                    "-\"\"\"Fresh-start additive ID model with no interaction or baseline-derived architecture.\"\"\"\n"
                    "+\"\"\"Fresh-start additive ID model with no interaction or inherited architecture.\"\"\"\n"
                )}},
            ])
            config = SystemConfig(
                workspace=str(root / "workspace"), data_dir=str(data),
                max_experiments=1, experiment_timeout_seconds=60, candidate_pool_size=2,
            )
            overseer = Overseer(config, llm)

            class DeterministicResearch:
                report_path = root / "screening_report.json"

                def ensure(self):
                    self.report_path.write_text("{}", encoding="utf-8")
                    return {}

                def evidence_ids(self, archive=None):
                    return ["screen:item_metadata"]

                def build_brief(self, archive):
                    return {
                        "screening_scope": "training_only_internal_temporal_holdout",
                        "ranked_feature_evidence": [{"evidence_id": "screen:item_metadata"}],
                    }

            overseer.research = DeterministicResearch()

            class DeterministicExperimentor:
                def run(self, sandbox, plan, *, attempt, timeout_seconds):
                    metrics = {
                        "GAUC": 0.7, "nDCG@5": 0.6, "primary": 0.65,
                        "users": 2, "rows": 4,
                        "classification": {
                            "accuracy": 0.75, "precision": 2 / 3,
                            "recall": 1.0, "f1": 0.8,
                        },
                        "ranking_diagnostics": {"MAP@5": 0.7},
                    }
                    (sandbox / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
                    (sandbox / "predictions_valid.npz").write_bytes(b"mock artifact")
                    return metrics, None

            overseer.experimentor = DeterministicExperimentor()
            result = overseer.run()
            self.assertEqual(result["stop_reason"], "experiment_cap")
            self.assertEqual(result["counted_experiments"], 1)
            experiment = config.run_dir / "experiment_1"
            self.assertTrue((experiment / "metrics.json").is_file())
            self.assertTrue((experiment / "predictions_valid.npz").is_file())
            self.assertEqual(len(llm.calls), 7)
            self.assertTrue(llm.calls[1]["payload"]["research_knowledge_base"])
            self.assertIn("classification", llm.calls[1]["payload"]["metric_catalog"])
            self.assertEqual(llm.calls[1]["payload"]["scored_metric_history"], [])
            seed_record = llm.calls[1]["payload"]["full_archive"][0]
            self.assertEqual(seed_record["status"], "seed")
            self.assertNotIn("primary", seed_record["metrics"])
            journal_record = overseer.journal.scored()[0]
            self.assertIn("model.py", journal_record["code_diff"])
            self.assertNotIn("config.json", journal_record["code_diff"])
            self.assertEqual(journal_record["metrics"]["classification"]["f1"], 0.8)
            self.assertEqual(journal_record["hypothesis_prediction"]["candidate_id"], "c1")
            patch_history = json.loads((experiment / "patch_history.json").read_text(encoding="utf-8"))
            self.assertEqual(len(patch_history), 2)
            self.assertTrue((experiment / "llm_events.jsonl").is_file())
            self.assertTrue((experiment / "attempt_summary.json").is_file())
            self.assertFalse((experiment / "failure.log").exists())


if __name__ == "__main__":
    unittest.main()
