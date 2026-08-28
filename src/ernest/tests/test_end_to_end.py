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
                    "seed", "embedding_dim", "learning_rate", "l2", "batch_size",
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
                {"hypotheses": [{
                    "text": "Document the FM seed without changing its behavior",
                    "parent_experiment_id": 0,
                    "scores": {"interestingness": 2, "novelty": 2, "feasibility": 10},
                    "rationale": "Pipeline smoke test",
                }]},
                {"accepted": True, "feedback": "feasible", "final_action": "not_applicable"},
                {"active_agents": ["model_designer"], "reasoning": "model-only", "contract": contract},
                {"patches": {"model.py": (
                    "--- model.py\n+++ model.py\n@@ -1,1 +1,1 @@\n"
                    "-\"\"\"Official-style NumPy Factorization Machine seed model.\"\"\"\n"
                    "+\"\"\"Official-style NumPy Factorization Machine seed model (documented).\"\"\"\n"
                )}},
            ])
            config = SystemConfig(
                workspace=str(root / "workspace"), data_dir=str(data),
                max_experiments=1, experiment_timeout_seconds=60,
            )
            overseer = Overseer(config, llm)

            class DeterministicExperimentor:
                def run(self, sandbox, plan, *, attempt, timeout_seconds):
                    metrics = {
                        "GAUC": 0.7, "nDCG@5": 0.6, "primary": 0.65,
                        "users": 2, "rows": 4,
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
            self.assertEqual(len(llm.calls), 4)
            self.assertTrue(llm.calls[0]["payload"]["research_knowledge_base"])
            journal_record = overseer.journal.scored()[0]
            self.assertIn("model.py", journal_record["code_diff"])
            self.assertNotIn("config.json", journal_record["code_diff"])


if __name__ == "__main__":
    unittest.main()
