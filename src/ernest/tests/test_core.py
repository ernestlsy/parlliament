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

from agentic_recsys.agents import EvolutionJudge, FeatureAnalyst, Orchestrator
from agentic_recsys.cli import _parser
from agentic_recsys.evaluation import METRIC_CATALOG, evaluate
from agentic_recsys.experimentor import Experimentor
from agentic_recsys.journal import Journal
from agentic_recsys.llm import LLMError, OpenAICompatibleClient, ScriptedLLMClient
from agentic_recsys.config import SystemConfig
from agentic_recsys.overseer import Overseer
from agentic_recsys.research import temporal_partition
from agentic_recsys.sandbox import GuardrailViolation, apply_agent_replacements, guarded_path
from agentic_recsys.schemas import (
    FailureKind, Hypothesis, HypothesisScores, InterfaceContract, JournalRecord, Mode,
    hypothesis_from_dict,
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
            for index, score in enumerate((0.5, 0.502, 0.501, 0.5015), 1):
                journal.append(record(index, score))
            self.assertFalse(journal.converged(0.002, 3))

    def test_convergence_compares_three_following_scores_to_one_anchor(self):
        with tempfile.TemporaryDirectory() as directory:
            journal = Journal(Path(directory) / "journal.jsonl")
            for index, score in enumerate((0.5, 0.501, 0.5015, 0.5019), 1):
                journal.append(record(index, score))
            self.assertTrue(journal.converged(0.002, 3))

    def test_any_following_score_reaching_anchor_threshold_prevents_stop(self):
        with tempfile.TemporaryDirectory() as directory:
            journal = Journal(Path(directory) / "journal.jsonl")
            for index, score in enumerate((0.5, 0.501, 0.5021, 0.5019), 1):
                journal.append(record(index, score))
            self.assertFalse(journal.converged(0.002, 3))


class SandboxTests(unittest.TestCase):
    def test_guarded_path_rejects_escape(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(GuardrailViolation):
                guarded_path(Path(directory), "../outside.py")

    def test_applies_scoped_complete_file_replacement(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "model.py").write_text("value = 1\n", encoding="utf-8")
            apply_agent_replacements(
                root, {"model.py": "value = 2\n"}, ("model.py",)
            )
            self.assertEqual((root / "model.py").read_text(encoding="utf-8"), "value = 2\n")

    def test_replacement_cannot_target_evaluator(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "model.py").write_text("value = 1\n", encoding="utf-8")
            with self.assertRaises(GuardrailViolation):
                apply_agent_replacements(
                    root,
                    {"model.py": "value = 2\n", "evaluation.py": "score = 1\n"},
                    ("model.py",),
                )

    def test_replacement_requires_every_owned_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "train.py").write_text("value = 1\n", encoding="utf-8")
            (root / "config.json").write_text('{"seed": 0}\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "missing complete files"):
                apply_agent_replacements(
                    root, {"train.py": "value = 2\n"}, ("train.py", "config.json")
                )

    def test_invalid_multi_file_replacement_is_not_partially_applied(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original_train = "value = 1\n"
            original_config = '{"seed": 0}\n'
            (root / "train.py").write_text(original_train, encoding="utf-8")
            (root / "config.json").write_text(original_config, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "invalid JSON"):
                apply_agent_replacements(
                    root,
                    {"train.py": "value = 2\n", "config.json": "{broken"},
                    ("train.py", "config.json"),
                )
            self.assertEqual((root / "train.py").read_text(encoding="utf-8"), original_train)
            self.assertEqual(
                (root / "config.json").read_text(encoding="utf-8"), original_config
            )

    def test_rejects_invalid_python_with_precise_location(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "model.py").write_text("value = 1\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, r"model\.py at line 1"):
                apply_agent_replacements(
                    root, {"model.py": "def broken(:\n"}, ("model.py",)
                )

    def test_rejects_complete_file_no_op(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            content = "value = 1\n"
            (root / "model.py").write_text(content, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "no changes"):
                apply_agent_replacements(root, {"model.py": content}, ("model.py",))

    def test_rejects_empty_and_non_string_complete_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "model.py").write_text("value = 1\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "cannot be empty"):
                apply_agent_replacements(root, {"model.py": "  "}, ("model.py",))
            with self.assertRaisesRegex(ValueError, "must be a string"):
                apply_agent_replacements(root, {"model.py": 3}, ("model.py",))


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

    def test_cli_seed_model_flag_defaults_and_selects_baseline(self):
        common = [
            "run", "--workspace", "workspace", "--data-dir", "data", "--model", "model",
        ]
        self.assertEqual(_parser().parse_args(common).seed_model, "simple")
        selected = _parser().parse_args(
            common + ["--seed-model", "kuairand-baseline"]
        )
        self.assertEqual(selected.seed_model, "kuairand-baseline")


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


class ResearchPlanningTests(unittest.TestCase):
    def test_probability_confidence_is_normalized_to_ten_point_scale(self):
        hypothesis = hypothesis_from_dict({
            "text": "normalize model confidence",
            "parent_experiment_id": 0,
            "scores": {"interestingness": 5, "novelty": 5, "feasibility": 5},
            "confidence": 0.62,
        })
        self.assertEqual(hypothesis.confidence, 6)

    def test_temporal_partition_uses_latest_dates_as_holdout(self):
        development, holdout = temporal_partition(
            [20220408, 20220409, 20220410, 20220411], 0.25
        )
        self.assertEqual(development, [20220408, 20220409, 20220410])
        self.assertEqual(holdout, [20220411])
        self.assertLess(max(development), min(holdout))

    def test_feature_analyst_repairs_unknown_evidence_reference(self):
        invalid = {
            "priorities": [{
                "evidence_ids": ["screen:not_real"],
                "finding": "unsupported",
                "recommended_action": "avoid",
            }]
        }
        valid = {
            "priorities": [{
                "evidence_ids": ["screen:item_metadata"],
                "finding": "stable lift",
                "recommended_action": "prioritize",
            }],
            "avoid": [],
            "metric_diagnosis": "top-list weakness",
        }
        llm = ScriptedLLMClient([invalid, valid])
        result = FeatureAnalyst(llm).analyze(
            research_brief={}, available_evidence_ids=["screen:item_metadata"]
        )
        self.assertEqual(result["priorities"][0]["finding"], "stable lift")
        self.assertIn("unavailable evidence", llm.calls[1]["payload"]["validation_feedback"])

    def test_tournament_candidate_contract_rejects_high_leakage(self):
        candidates = []
        for index, risk in enumerate(("low", "high"), 1):
            candidates.append({
                "candidate_id": f"c{index}",
                "text": f"candidate {index}",
                "parent_experiment_id": 0,
                "scores": {"interestingness": 5, "novelty": 5, "feasibility": 5},
                "rationale": "test",
                "evidence_ids": ["screen:item_metadata"],
                "exact_ablation": f"change {index}",
                "expected_effect": {"GAUC": "up", "nDCG@5": "up"},
                "expected_primary_gain": 0.003,
                "confidence": 7,
                "leakage_risk": risk,
                "runtime_risk": "low",
                "active_components": ["feature_engineer"],
            })
        valid = dict(candidates[1])
        valid["leakage_risk"] = "low"
        llm = ScriptedLLMClient([
            {"candidates": candidates},
            {"candidates": [candidates[0], valid]},
        ])
        judge = EvolutionJudge(llm)
        result = judge.generate_candidates(
            mode=Mode.DRAFT,
            generation=1,
            archive=[],
            reference_snapshots=[{"experiment_id": 0, "files": {}, "metrics": {}}],
            candidate_count=2,
            research_brief={},
            analyst_assessment={},
            available_evidence_ids=["screen:item_metadata"],
        )
        self.assertEqual(len(result), 2)
        self.assertIn("high leakage risk", llm.calls[1]["payload"]["validation_feedback"])

    def test_orchestrator_instructions_and_schema_reach_feature_engineer(self):
        instruction = {
            "objective": "Derive cyclic hour from the raw request-time field",
            "required_changes": [
                "Read hourmin and derive hour as floor(hourmin / 100)",
                "Add hour_sin and hour_cos to data.py",
            ],
            "preserve": ["Canonical validation row order"],
            "coordination_notes": ["Expose the two numeric columns to model.py"],
        }
        contract = {
            "data_output": {"description": "categorical fields plus cyclic hour"},
            "config_keys": ["seed"],
            "model_input": {"description": "encoded feature rows"},
        }
        seed_dir = Path(__file__).parents[1] / "agentic_recsys" / "seed"
        schema = {
            "raw_sources": {
                "log_standard_4_08_to_4_21_pure.csv": [{
                    "name": "hourmin", "status": "eligible", "reason": "available",
                }]
            },
            "derived_features": {
                "hour": {
                    "source_columns": ["hourmin"],
                    "recipe": "integer hour = floor(numeric hourmin / 100)",
                }
            },
        }
        llm = ScriptedLLMClient([{
            "active_agents": ["feature_engineer"],
            "agent_instructions": {"feature_engineer": instruction},
            "reasoning": "data-only change",
            "contract": contract,
        }, {
            "files": {"data.py": (seed_dir / "data.py").read_text(encoding="utf-8")},
        }])
        orchestrator = Orchestrator(llm)
        hypothesis = Hypothesis(
            text="Replace categorical hour with cyclic hour",
            parent_experiment_id=0,
            scores=HypothesisScores(7, 7, 8),
        )
        plan = orchestrator.plan(hypothesis, seed_dir, dataset_feature_schema=schema)
        orchestrator.generate_file_replacements(
            agent="feature_engineer",
            hypothesis=hypothesis,
            plan=plan,
            sandbox=seed_dir,
            dataset_feature_schema=schema,
        )
        self.assertEqual(plan.agent_instructions["feature_engineer"], instruction)
        self.assertEqual(llm.calls[0]["payload"]["dataset_feature_schema"], schema)
        self.assertEqual(llm.calls[1]["payload"]["dataset_feature_schema"], schema)
        self.assertEqual(llm.calls[1]["payload"]["agent_instruction"], instruction)

    def test_candidate_knowledge_id_is_reclassified_as_literature(self):
        llm = ScriptedLLMClient([{"candidates": [{
            "candidate_id": "c1",
            "text": "Change only the ranking loss",
            "parent_experiment_id": 0,
            "scores": {"interestingness": 7, "novelty": 6, "feasibility": 8},
            "rationale": "Measured screening and metric guidance support the test",
            "evidence_ids": [
                "screen:item_metadata", "evaluation.within_user_metrics",
            ],
            "exact_ablation": "replace BCE with one fixed pairwise ranking loss",
            "expected_effect": {"GAUC": "small gain", "nDCG@5": "possible gain"},
            "expected_primary_gain": 0.002,
            "confidence": 6,
            "leakage_risk": "low",
            "runtime_risk": "low",
            "active_components": ["trainer"],
            "literature_document_ids": [],
        }]}])
        judge = EvolutionJudge(llm, knowledge_documents=[{
            "id": "evaluation.within_user_metrics",
            "title": "Within-user metrics",
            "content": "Metric guidance",
        }])
        result = judge.generate_candidates(
            mode=Mode.DRAFT,
            generation=1,
            archive=[],
            reference_snapshots=[{"experiment_id": 0, "files": {}, "metrics": {}}],
            candidate_count=1,
            research_brief={},
            analyst_assessment={},
            available_evidence_ids=["screen:item_metadata"],
        )
        self.assertEqual(result[0].evidence_ids, ["screen:item_metadata"])
        self.assertEqual(
            result[0].literature_document_ids, ["evaluation.within_user_metrics"]
        )
        self.assertEqual(
            llm.calls[0]["payload"]["available_literature_document_ids"],
            ["evaluation.within_user_metrics"],
        )


class ReferenceRefreshTests(unittest.TestCase):
    def test_seed_model_selects_requested_parent_zero_scaffold(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_dir = root / "data"
            data_dir.mkdir()
            overseer = Overseer(SystemConfig(
                workspace=str(root / "workspace"),
                data_dir=str(data_dir),
                seed_model="kuairand-baseline",
            ), ScriptedLLMClient([]))
            self.assertEqual(overseer.seed_dir.name, "seed_kuairand_baseline")
            seed = overseer._archive()[0]
            self.assertEqual(seed["seed_model"], "kuairand-baseline")
            self.assertIn("Factorization Machine", seed["hypothesis_text"])

    def test_run_cannot_resume_with_a_different_seed_model(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_dir = root / "data"
            data_dir.mkdir()
            simple = Overseer(SystemConfig(
                workspace=str(root / "workspace"),
                data_dir=str(data_dir),
                run_name="fixed-seed-run",
                seed_model="simple",
            ), ScriptedLLMClient([]))
            simple.initialize()
            changed = Overseer(SystemConfig(
                workspace=str(root / "workspace"),
                data_dir=str(data_dir),
                run_name="fixed-seed-run",
                seed_model="kuairand-baseline",
            ), ScriptedLLMClient([]))
            with self.assertRaisesRegex(ValueError, "cannot resume"):
                changed.initialize()

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
    def test_read_timeout_is_retried(self, urlopen):
        response = mock.MagicMock()
        response.__enter__.return_value = response
        response.read.return_value = json.dumps({
            "output": [{
                "type": "message",
                "content": [{"type": "output_text", "text": '{"ok": true}'}],
            }],
        }).encode("utf-8")
        urlopen.side_effect = [TimeoutError("read timed out"), response]
        client = OpenAICompatibleClient(
            "test-model", api_key="secret", max_retries=1, retry_backoff_seconds=0,
        )
        self.assertEqual(
            client.complete_json(role="judge", system="return JSON", payload={}),
            {"ok": True},
        )
        self.assertEqual(urlopen.call_count, 2)

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
    def test_builder_web_search_request_is_required_and_captures_citations(self, urlopen):
        response = mock.MagicMock()
        response.__enter__.return_value = response
        response.read.return_value = json.dumps({
            "output": [{
                "type": "web_search_call",
                "id": "ws_test",
                "status": "completed",
                "action": {
                    "type": "search",
                    "query": "recommender ranking paper",
                    "sources": [{"url": "https://example.org/paper"}],
                },
            }, {
                "type": "message",
                "content": [{
                    "type": "output_text",
                    "text": '{"ok": true}',
                    "annotations": [{
                        "type": "url_citation",
                        "url": "https://example.org/paper",
                        "title": "Primary paper",
                    }],
                }],
            }],
        }).encode("utf-8")
        urlopen.return_value = response
        client = OpenAICompatibleClient(
            "test-model", api_key="secret", web_search=True,
            web_search_context_size="high",
        )
        result = client.complete_json(role="knowledge_card_writer", system="research", payload={})
        request = urlopen.call_args.args[0]
        body = json.loads(request.data)
        self.assertEqual(
            body["tools"], [{"type": "web_search", "search_context_size": "high"}]
        )
        self.assertEqual(body["tool_choice"], "required")
        self.assertEqual(body["include"], ["web_search_call.action.sources"])
        self.assertNotIn("text", body)
        self.assertIn("valid JSON object", body["input"])
        self.assertEqual(result["_web_search"]["calls"][0]["id"], "ws_test")
        self.assertEqual(result["_web_search"]["citations"], [{
            "title": "Primary paper", "url": "https://example.org/paper",
        }])

    def test_hosted_web_search_rejects_chat_completions_mode(self):
        with self.assertRaisesRegex(ValueError, "requires the Responses API"):
            OpenAICompatibleClient(
                "test-model", api_key="secret", api_mode="chat", web_search=True,
            )

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
