import io
import importlib.util
import json
import random
import shutil
import tempfile
import unittest
import urllib.error
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from unittest import mock

from agentic_recsys.agents import EvolutionJudge
from agentic_recsys.evaluation import METRIC_CATALOG, evaluate
from agentic_recsys.experimentor import Experimentor
from agentic_recsys.journal import Journal
from agentic_recsys.llm import LLMError, OpenAICompatibleClient, ScriptedLLMClient
from agentic_recsys.config import SystemConfig
from agentic_recsys.overseer import Overseer
from agentic_recsys.sandbox import GuardrailViolation, apply_unified_diff, guarded_path
from agentic_recsys.schemas import (
    FailureKind, Hypothesis, HypothesisScores, InterfaceContract, JournalRecord, Mode,
)


def record(experiment_id, primary, status="scored", sandbox="x"):
    return JournalRecord(
        attempt_id=f"a{experiment_id}",
        experiment_id=experiment_id if status == "scored" else None,
        generation=experiment_id,
        parent_experiment_id=max(0, experiment_id - 1),
        hypothesis_text=f"h{experiment_id}",
        hypothesis_scores={"interestingness": 5, "novelty": 5, "feasibility": 5},
        mode="draft", code_diff={}, config_diff="", active_sub_agents=[],
        metrics={"primary": primary} if status == "scored" else {},
        status=status, failure_reason=None if status == "scored" else "boom",
        failure_stage=None if status == "scored" else "test",
        consultant_rounds=1, sandbox=sandbox, created_at=datetime.now(timezone.utc).isoformat(),
    )


class EvaluationTests(unittest.TestCase):
    def test_perfect_ranking_and_zero_positive_user(self):
        result = evaluate(
            ["a", "a", "b", "b"],
            [1, 0, 0, 0],
            [1.0, 0.0, 1.0, 0.0],
        )
        self.assertEqual(result["GAUC"], 1.0)
        self.assertEqual(result["nDCG@5"], 0.5)
        self.assertEqual(result["primary"], 0.75)

    def test_tie_corrected_auc(self):
        result = evaluate([1, 1], [1, 0], [0.2, 0.2])
        self.assertEqual(result["GAUC"], 0.5)

    def test_classification_and_ranking_diagnostics(self):
        result = evaluate(
            ["a", "a", "b", "b"],
            [1, 0, 1, 0],
            [0.9, 0.8, 0.7, 0.1],
        )
        classification = result["classification"]
        self.assertEqual(classification["threshold"], 0.7)
        self.assertEqual(classification["confusion_matrix"], {
            "true_positive": 2,
            "false_positive": 1,
            "true_negative": 1,
            "false_negative": 0,
        })
        self.assertAlmostEqual(classification["accuracy"], 0.75)
        self.assertAlmostEqual(classification["precision"], 2 / 3)
        self.assertEqual(classification["recall"], 1.0)
        self.assertAlmostEqual(classification["f1"], 0.8)
        ranking = result["ranking_diagnostics"]
        self.assertEqual(ranking["Recall@5"], 1.0)
        self.assertEqual(ranking["HitRate@5"], 1.0)
        self.assertEqual(ranking["MRR@5"], 1.0)

    def test_all_negative_classification_is_well_defined(self):
        result = evaluate(["a", "a"], [0, 0], [0.2, 0.1])
        classification = result["classification"]
        self.assertIsNone(classification["threshold"])
        self.assertEqual(classification["accuracy"], 1.0)
        self.assertEqual(classification["precision"], 0.0)
        self.assertEqual(classification["recall"], 0.0)
        self.assertEqual(result["ranking_diagnostics"]["average_precision"], 0.0)

    def test_metric_catalog_covers_every_diagnostic(self):
        result = evaluate(["a", "a"], [1, 0], [0.9, 0.1])
        classification_fields = set(result["classification"]) - {"threshold_strategy"}
        self.assertEqual(
            classification_fields,
            set(METRIC_CATALOG["classification"]["metrics"]),
        )
        self.assertEqual(
            set(result["ranking_diagnostics"]),
            set(METRIC_CATALOG["ranking_diagnostics"]),
        )
        self.assertEqual(
            set(result["data_diagnostics"]),
            set(METRIC_CATALOG["data_diagnostics"]),
        )

    def test_rejects_non_finite_scores(self):
        with self.assertRaises(ValueError):
            evaluate([1], [1], [float("nan")])

    def test_matches_starter_kit_evaluator(self):
        path = Path(__file__).parents[2] / "kuairand-starter-kit" / "evaluate.py"
        spec = importlib.util.spec_from_file_location("starter_evaluate", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        rng = random.Random(7)
        users = [index // 7 for index in range(140)]
        labels = [rng.randrange(2) for _ in users]
        scores = [rng.random() for _ in users]
        expected = module.evaluate(users, labels, scores)
        actual = evaluate(users, labels, scores)
        for key in ("GAUC", "nDCG@5", "primary"):
            self.assertAlmostEqual(actual[key], expected[key], places=14)


class JournalTests(unittest.TestCase):
    def test_abandoned_does_not_consume_id(self):
        with tempfile.TemporaryDirectory() as directory:
            journal = Journal(Path(directory) / "journal.jsonl")
            journal.append(record(1, 0.5))
            journal.append(record(999, 0, "abandoned"))
            self.assertEqual(journal.next_experiment_id(), 2)

    def test_convergence_is_strict_at_epsilon(self):
        with tempfile.TemporaryDirectory() as directory:
            journal = Journal(Path(directory) / "journal.jsonl")
            for index, score in enumerate((0.5, 0.501, 0.502), 1):
                journal.append(record(index, score))
            self.assertFalse(journal.converged(0.002, 3))
            journal.append(record(4, 0.5025))
            self.assertTrue(journal.converged(0.002, 3))


class SandboxTests(unittest.TestCase):
    def test_guarded_path_rejects_escape(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(GuardrailViolation):
                guarded_path(Path(directory), "../outside.py")

    def test_applies_scoped_unified_diff(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "model.py").write_text("one\ntwo\n", encoding="utf-8")
            apply_unified_diff(
                root,
                "model.py",
                "--- model.py\n+++ model.py\n@@ -1,2 +1,2 @@\n one\n-two\n+three\n",
            )
            self.assertEqual((root / "model.py").read_text(encoding="utf-8"), "one\nthree\n")

    def test_patch_cannot_target_evaluator(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(GuardrailViolation):
                apply_unified_diff(Path(directory), "evaluation.py", "")

    def test_tolerates_llm_trailing_hunk_marker(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "model.py").write_text("old\n", encoding="utf-8")
            apply_unified_diff(
                root,
                "model.py",
                "--- model.py\n+++ model.py\n@@ -1,1 +1,1 @@\n-old\n+new\n@@\n",
            )
            self.assertEqual((root / "model.py").read_text(encoding="utf-8"), "new\n")


class FailureClassificationTests(unittest.TestCase):
    def test_resource_and_contract_classification(self):
        resource = Experimentor.classify_process_failure("CUDA out of memory", 1, 1)
        contract = Experimentor.classify_process_failure("ValueError: shape mismatch in model.py", 1, 1)
        self.assertEqual(resource.kind, FailureKind.RESOURCE)
        self.assertEqual(contract.kind, FailureKind.CONTRACT_USAGE)
        self.assertIn("model_designer", contract.responsible_agents)


class ContractTests(unittest.TestCase):
    def test_agents_cannot_replace_execution_command(self):
        with self.assertRaises(ValueError):
            InterfaceContract.from_dict({
                "data_output": {"features": "array"},
                "config_keys": ["seed"],
                "model_input": {"features": "array"},
                "train_command": ["{python}", "-c", "print('not allowed')"],
            })


class EvolutionJudgeMetricAccessTests(unittest.TestCase):
    def test_complete_nested_metrics_are_sent_to_judge(self):
        metrics = {
            "GAUC": 0.7,
            "nDCG@5": 0.6,
            "primary": 0.65,
            "classification": {
                "accuracy": 0.75,
                "precision": 0.8,
                "recall": 0.5,
                "f1": 0.615,
                "confusion_matrix": {
                    "true_positive": 10, "false_positive": 2,
                    "true_negative": 20, "false_negative": 10,
                },
            },
            "ranking_diagnostics": {"MAP@5": 0.55, "MRR@5": 0.61},
            "data_diagnostics": {"positive_rate": 0.4, "score_std": 0.2},
        }
        archive = [{
            "experiment_id": 1,
            "generation": 1,
            "parent_experiment_id": 0,
            "hypothesis_text": "first counted experiment",
            "status": "scored",
            "metrics": metrics,
        }]
        llm = ScriptedLLMClient([{"hypotheses": [{
            "text": "improve top-list recall",
            "parent_experiment_id": 1,
            "scores": {"interestingness": 8, "novelty": 7, "feasibility": 8},
        }]}])
        judge = EvolutionJudge(llm, metric_catalog=METRIC_CATALOG)
        judge.propose(
            mode=Mode.IMPROVE,
            generation=2,
            archive=archive,
            reference_snapshots=[{
                "experiment_id": 1,
                "generation": 1,
                "metrics": metrics,
                "files": {},
            }],
            count=1,
        )
        payload = llm.calls[0]["payload"]
        self.assertEqual(payload["scored_metric_history"][0]["metrics"], metrics)
        self.assertEqual(payload["full_archive"][0]["metrics"], metrics)
        self.assertEqual(payload["metric_catalog"], METRIC_CATALOG)

    def test_invalid_parent_reference_is_returned_for_repair(self):
        invalid = {"hypotheses": [{
            "text": "invalid self reference",
            "parent_experiment_id": 1,
            "scores": {"interestingness": 5, "novelty": 5, "feasibility": 5},
        }]}
        valid = {"hypotheses": [{
            "text": "branch from the available seed",
            "parent_experiment_id": 0,
            "scores": {"interestingness": 5, "novelty": 5, "feasibility": 5},
        }]}
        llm = ScriptedLLMClient([invalid, valid])
        judge = EvolutionJudge(llm, metric_catalog=METRIC_CATALOG)
        proposals = judge.propose(
            mode=Mode.DRAFT,
            generation=1,
            archive=[],
            reference_snapshots=[{
                "experiment_id": 0,
                "generation": 0,
                "metrics": {"availability": "unscored"},
                "files": {},
            }],
            count=1,
        )
        self.assertEqual(proposals[0].parent_experiment_id, 0)
        self.assertEqual(len(llm.calls), 2)
        self.assertIn("available parents are [0]", llm.calls[1]["payload"]["validation_feedback"])


class ReferenceRefreshTests(unittest.TestCase):
    def test_newly_scored_experiment_becomes_replacement_parent(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_dir = root / "data"
            data_dir.mkdir()
            overseer = Overseer(SystemConfig(
                workspace=str(root / "workspace"), data_dir=str(data_dir), max_experiments=2,
            ), ScriptedLLMClient([]))
            overseer.initialize()
            experiment_dir = overseer.config.run_dir / "experiment_1"
            shutil.copytree(overseer.seed_dir, experiment_dir)
            overseer.journal.append(record(1, 0.6, sandbox=str(experiment_dir)))
            _, _, snapshots, parent_ids = overseer._current_judge_context(Mode.DRAFT)
            self.assertEqual(parent_ids, [0, 1])
            self.assertEqual([item["experiment_id"] for item in snapshots], [0, 1])


class HTTPClientTests(unittest.TestCase):
    @mock.patch("agentic_recsys.llm.urllib.request.urlopen")
    def test_openai_auto_mode_uses_responses_api(self, urlopen):
        response = mock.MagicMock()
        response.__enter__.return_value = response
        response.read.return_value = json.dumps({
            "output": [{
                "type": "message",
                "content": [{"type": "output_text", "text": '{"ok": true}'}],
            }],
        }).encode("utf-8")
        urlopen.return_value = response
        client = OpenAICompatibleClient("test-model", api_key="secret")
        self.assertEqual(
            client.complete_json(role="judge", system="follow the task", payload={"x": 1}),
            {"ok": True},
        )
        request = urlopen.call_args.args[0]
        body = json.loads(request.data)
        self.assertTrue(request.full_url.endswith("/responses"))
        self.assertEqual(body["text"]["format"], {"type": "json_object"})
        self.assertIn("JSON", body["input"])
        self.assertIn('"x": 1', body["input"])
        self.assertNotIn("temperature", body)

    @mock.patch("agentic_recsys.llm.urllib.request.urlopen")
    def test_chat_mode_omits_unsupported_temperature(self, urlopen):
        response = mock.MagicMock()
        response.__enter__.return_value = response
        response.read.return_value = json.dumps({
            "choices": [{"message": {"content": '{"ok": true}'}}],
        }).encode("utf-8")
        urlopen.return_value = response
        client = OpenAICompatibleClient(
            "test-model", api_key="secret", api_mode="chat",
        )
        client.complete_json(role="judge", system="return JSON", payload={})
        request = urlopen.call_args.args[0]
        body = json.loads(request.data)
        self.assertTrue(request.full_url.endswith("/chat/completions"))
        self.assertNotIn("temperature", body)

    @mock.patch("agentic_recsys.llm.urllib.request.urlopen")
    def test_http_error_includes_provider_response(self, urlopen):
        urlopen.side_effect = urllib.error.HTTPError(
            "https://api.openai.com/v1/responses",
            400,
            "Bad Request",
            {"x-request-id": "req_test"},
            io.BytesIO(b'{"error":{"message":"unsupported field"}}'),
        )
        client = OpenAICompatibleClient("test-model", api_key="secret")
        with self.assertRaises(LLMError) as caught:
            client.complete_json(role="judge", system="return JSON", payload={})
        message = str(caught.exception)
        self.assertIn("HTTP 400", message)
        self.assertIn("req_test", message)
        self.assertIn("unsupported field", message)


class AbandonmentLoggingTests(unittest.TestCase):
    def test_invalid_orchestrator_responses_leave_specific_logs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_dir = root / "data"
            data_dir.mkdir()
            invalid_plan = {
                "active_agents": ["model_designer"],
                "reasoning": "incomplete on purpose",
                "contract": {},
            }
            llm = ScriptedLLMClient([invalid_plan, invalid_plan, invalid_plan])
            overseer = Overseer(SystemConfig(
                workspace=str(root / "workspace"),
                data_dir=str(data_dir),
                max_experiments=1,
            ), llm)
            overseer.initialize()
            hypothesis = Hypothesis(
                text="exercise invalid-plan diagnostics",
                parent_experiment_id=0,
                scores=HypothesisScores(5, 5, 5),
            )
            record = overseer._run_hypothesis(
                hypothesis,
                generation=1,
                mode=Mode.DRAFT,
                consultant_rounds=1,
                caveat=None,
            )
            sandbox = Path(record.sandbox)
            self.assertEqual(record.status, "abandoned")
            self.assertEqual(record.failure_stage, "orchestrator_plan")
            self.assertIn("invalid plans three times", record.failure_reason)
            self.assertTrue((sandbox / "failure.log").is_file())
            summary = json.loads((sandbox / "attempt_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["failure_stage"], "orchestrator_plan")
            self.assertIn("invalid plans three times", summary["failure_reason"])
            events = [
                json.loads(line)
                for line in (sandbox / "llm_events.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(len(events), 3)
            self.assertTrue(all(event["status"] == "success" for event in events))
            self.assertTrue(all("response" in event for event in events))


if __name__ == "__main__":
    unittest.main()
