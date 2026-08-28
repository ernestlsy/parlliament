import importlib.util
import random
import tempfile
import unittest
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from agentic_recsys.evaluation import evaluate
from agentic_recsys.experimentor import Experimentor
from agentic_recsys.journal import Journal
from agentic_recsys.sandbox import GuardrailViolation, apply_unified_diff, guarded_path
from agentic_recsys.schemas import FailureKind, InterfaceContract, JournalRecord


def record(experiment_id, primary, status="scored"):
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
        consultant_rounds=1, sandbox="x", created_at=datetime.now(timezone.utc).isoformat(),
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


if __name__ == "__main__":
    unittest.main()
