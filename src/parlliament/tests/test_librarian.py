import json
import shutil
import tempfile
import unittest
from pathlib import Path

from parlliament.config import SystemConfig
from parlliament.journal import Journal
from parlliament.librarian import (
    CatalogRetriever,
    KnowledgeCatalog,
    Librarian,
    ResearchRequest,
    RetrievalSettings,
)
from parlliament.llm import ScriptedLLMClient
from parlliament.overseer import Overseer
from parlliament.schemas import Mode


KNOWLEDGE_ROOT = Path(__file__).parents[1] / "parlliament" / "knowledge"


def retrieval_context():
    return {
        "available_features": [
            "user_id", "item_id", "categorical_fields", "interaction_logs", "timestamps",
            "binary_labels", "engagement_labels", "scores", "request_context",
        ],
        "metric_weaknesses": "nDCG@5 and rare-user ranking are weak",
        "current_architecture": "additive user and item model",
        "tested_hypotheses": "neutral seed",
        "experiment_timeout_seconds": 900,
    }


class KnowledgeCatalogTests(unittest.TestCase):
    def test_shipped_catalog_and_hashes_are_valid(self):
        catalog = KnowledgeCatalog(KNOWLEDGE_ROOT)
        self.assertGreaterEqual(len(catalog.entries), 11)
        fixed_ids = {item["id"] for item in catalog.fixed_documents()}
        self.assertEqual(len(fixed_ids), 6)
        self.assertIn("task.starter_kit_empirical_priors", fixed_ids)
        self.assertEqual(len(catalog.by_id), len(catalog.entries))
        self.assertEqual(
            sum(bool(item.get("system_owned")) for item in catalog.entries), 12
        )

    def test_catalog_rejects_parent_traversal(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "knowledge"
            root.mkdir()
            (root / "catalog.jsonl").write_text(json.dumps({
                "id": "bad.path",
                "path": "../outside.md",
                "title": "bad",
                "category": "task",
                "tags": [],
                "summary": "bad",
                "use_when": [],
                "avoid_when": [],
                "required_features": ["none"],
                "metrics": [],
            }) + "\n", encoding="utf-8")
            (root / "manifest.json").write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "unsafe knowledge-card path"):
                KnowledgeCatalog(root)

    def test_catalog_detects_document_hash_change(self):
        with tempfile.TemporaryDirectory() as directory:
            copied = Path(directory) / "knowledge"
            shutil.copytree(KNOWLEDGE_ROOT, copied)
            card = copied / "00_task" / "leakage_policy.md"
            card.write_text(card.read_text(encoding="utf-8") + "\nchanged\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "document hash mismatch"):
                KnowledgeCatalog(copied)

    def test_catalog_hashes_are_portable_across_crlf_checkouts(self):
        initial_card_count = len(KnowledgeCatalog(KNOWLEDGE_ROOT).entries)
        with tempfile.TemporaryDirectory() as directory:
            copied = Path(directory) / "knowledge"
            shutil.copytree(KNOWLEDGE_ROOT, copied)
            text_files = [copied / "catalog.jsonl", *copied.rglob("*.md")]
            for path in text_files:
                normalized = path.read_bytes().replace(b"\r\n", b"\n")
                path.write_bytes(normalized.replace(b"\n", b"\r\n"))
            catalog = KnowledgeCatalog(copied)
            self.assertEqual(len(catalog.entries), initial_card_count)


class DeterministicRetrievalTests(unittest.TestCase):
    def test_tfidf_mmr_is_deterministic_for_fixed_catalog(self):
        catalog = KnowledgeCatalog(KNOWLEDGE_ROOT)
        retriever = CatalogRetriever(catalog)
        request = ResearchRequest(
            query="interpret within-user ranking and top-five diagnostics",
            purpose="understand the fixed evaluation objective",
            categories=("task", "evaluation"),
            preferred_tags=("within-user", "diagnostics"),
            metrics_of_interest=("GAUC", "nDCG@5"),
            max_documents=6,
        )
        first = retriever.search(request, retrieval_context(), limit=10)
        second = retriever.search(request, retrieval_context(), limit=10)
        self.assertEqual(first, second)
        identifiers = [item["id"] for item in first]
        self.assertIn("evaluation.within_user_metrics", identifiers)
        self.assertIn("task.kuairand_ranking", identifiers)


class HybridLibrarianTests(unittest.TestCase):
    def test_hybrid_selection_repairs_hallucinated_id_and_writes_audit(self):
        llm = ScriptedLLMClient([
            {"alternative_queries": [
                "Bayesian personalized ranking sampled negatives",
                "pairwise implicit-feedback ranking optimization",
            ]},
            {"selected_document_ids": ["invented.document"]},
            {"selected_document_ids": [
                "task.kuairand_ranking",
                "task.leakage_policy",
                "evaluation.within_user_metrics",
            ]},
        ])
        librarian = Librarian(
            llm,
            KNOWLEDGE_ROOT,
            RetrievalSettings(max_final_documents=3, character_budget=1_200),
        )
        request = ResearchRequest(
            query="interpret ranking metrics while respecting leakage constraints",
            purpose="retrieve the fixed task, evaluation, and safety context",
            categories=(),
            preferred_tags=("within-user", "leakage"),
            metrics_of_interest=("GAUC", "nDCG@5"),
            max_documents=3,
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            journal = Journal(root / "journal.jsonl")
            result = librarian.retrieve(
                [request], context=retrieval_context(), round_number=1
            )
            librarian.write_audit(root / "literature", [result])
            self.assertEqual(len(llm.calls), 3)
            self.assertEqual(
                llm.calls[-1]["payload"]["validation_feedback"],
                "Response validation failed: ValueError: selected_document_ids requires 3-3 IDs",
            )
            self.assertEqual(len(result["documents"]), 3)
            self.assertLessEqual(
                sum(len(item["content"]) for item in result["documents"]), 1_200
            )
            self.assertEqual(journal.records(), [])
            expected = {
                "research_requests.json", "deterministic_candidates.json",
                "llm_expanded_queries.json", "llm_retrieval_candidates.json",
                "merged_candidates.json", "selected_document_ids.json",
                "retrieval_manifest.json",
            }
            self.assertEqual(
                {path.name for path in (root / "literature").iterdir()}, expected
            )
            manifest = json.loads(
                (root / "literature" / "retrieval_manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["retrieval_rounds"], 1)
            self.assertLessEqual(manifest["total_fetched_characters"], 1_200)

    def test_tournament_retrieves_literature_before_candidate_generation(self):
        llm = ScriptedLLMClient([
            {"priorities": [{
                "evidence_ids": ["screen:item_metadata"],
                "finding": "ranking needs a stable training change",
                "recommended_action": "review optimization literature",
            }], "avoid": [], "metric_diagnosis": "nDCG@5 is weak"},
            {"research_requests": [{
                "query": "fixed ranking objective and leakage boundary",
                "purpose": "confirm task and safety constraints before proposing a change",
                "categories": ["task"],
                "preferred_tags": ["within-user", "leakage"],
                "metrics_of_interest": ["GAUC", "nDCG@5"],
                "max_documents": 2,
            }]},
            {"alternative_queries": ["impression-time causal ranking constraints"]},
            {"selected_document_ids": [
                "task.kuairand_ranking", "task.leakage_policy",
            ]},
            {"research_requests": []},
            {"candidates": [{
                "candidate_id": "c1",
                "text": "Change only L2 regularization",
                "parent_experiment_id": 0,
                "scores": {"interestingness": 7, "novelty": 5, "feasibility": 9},
                "rationale": "Literature supports a bounded stability test",
                "evidence_ids": ["screen:item_metadata"],
                "exact_ablation": "change only L2 from 1e-6 to 1e-5",
                "expected_effect": {"GAUC": "small gain", "nDCG@5": "small gain"},
                "expected_primary_gain": 0.002,
                "confidence": 7,
                "leakage_risk": "low",
                "runtime_risk": "low",
                "active_components": ["training"],
                "literature_document_ids": ["task.kuairand_ranking"],
            }, {
                "candidate_id": "c2",
                "text": "Change only negative sampling",
                "parent_experiment_id": 0,
                "scores": {"interestingness": 6, "novelty": 6, "feasibility": 7},
                "rationale": "Alternative ranking-aware training test",
                "evidence_ids": ["screen:item_metadata"],
                "exact_ablation": "change only the negative sampler",
                "expected_effect": {"GAUC": "uncertain", "nDCG@5": "possible gain"},
                "expected_primary_gain": 0.001,
                "confidence": 5,
                "leakage_risk": "low",
                "runtime_risk": "medium",
                "active_components": ["training"],
                "literature_document_ids": ["task.leakage_policy"],
            }]},
            {"ranking": [
                {"candidate_id": "c1", "rank": 1, "utility_score": 8, "rationale": "safer"},
                {"candidate_id": "c2", "rank": 2, "utility_score": 5, "rationale": "riskier"},
            ]},
            {"winner_candidate_id": "c1", "selection_rationale": "best bounded gain"},
        ])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "data"
            data.mkdir()
            config = SystemConfig(
                workspace=str(root / "workspace"),
                data_dir=str(data),
                candidate_pool_size=2,
                literature_max_documents=4,
                literature_character_budget=4_000,
            )
            overseer = Overseer(config, llm)

            class DeterministicResearch:
                report_path = root / "screening_report.json"

                def build_brief(self, archive):
                    return {
                        "ranked_feature_evidence": [{
                            "evidence_id": "screen:item_metadata",
                            "feature_group": "item_metadata",
                            "fields": ["author_id"],
                        }]
                    }

                def evidence_ids(self, archive=None):
                    return ["screen:item_metadata"]

            overseer.research = DeterministicResearch()
            overseer.initialize()
            archive = overseer._archive()
            snapshots = [overseer._snapshot(archive[0])]
            winner, rounds, caveat = overseer._run_tournament(
                generation=1, mode=Mode.DRAFT, archive=archive, snapshots=snapshots
            )
            self.assertEqual(winner.candidate_id, "c1")
            self.assertEqual(winner.literature_document_ids, ["task.kuairand_ranking"])
            self.assertEqual((rounds, caveat), (1, None))
            self.assertEqual(overseer.journal.records(), [])
            literature = config.run_dir / "planning" / "generation_1" / "literature"
            self.assertTrue((literature / "retrieval_manifest.json").is_file())
            candidate_call = next(
                call for call in llm.calls if call["role"] == "evolution_judge_candidates"
            )
            fixed_document_ids = {
                item["id"] for item in overseer.judge.knowledge_documents
            }
            self.assertEqual(
                set(candidate_call["payload"]["available_literature_document_ids"]),
                fixed_document_ids | {"task.kuairand_ranking", "task.leakage_policy"},
            )
            self.assertEqual(len(candidate_call["payload"]["retrieved_literature"]), 2)


if __name__ == "__main__":
    unittest.main()
