import shutil
import tempfile
import unittest
from pathlib import Path

from agentic_recsys.knowledge_builder import (
    GENERATED_CATEGORIES,
    SYSTEM_OWNED_CATEGORIES,
    KnowledgeBaseBuilder,
    REQUIRED_HEADINGS,
)
from agentic_recsys.librarian import KnowledgeCatalog
from agentic_recsys.llm import ScriptedLLMClient


KNOWLEDGE_ROOT = Path(__file__).parents[1] / "agentic_recsys" / "knowledge"


def plan(slug="gradient_conflict", title="Multitask gradient-conflict management"):
    return {"cards": [{
        "slug": slug,
        "title": title,
        "tags": ["multitask", "optimization", "gradients"],
        "summary": "Manages conflicting task gradients while retaining shared recommender representations.",
        "use_when": ["auxiliary engagement tasks disagree during optimization"],
        "avoid_when": ["the recommender has only one supervised objective"],
        "required_features": ["multiple engagement labels"],
        "metrics": ["GAUC", "nDCG@5"],
        "method_family": slug,
    }]}


def card(title="Multitask gradient-conflict management"):
    sections = []
    for heading in REQUIRED_HEADINGS:
        sections.append(
            f"## {heading}\n\n"
            "Use train-only evidence, isolate one change, inspect both within-user ranking metrics, "
            "and retain a conservative fallback configuration. This section states assumptions, "
            "implementation choices, diagnostic signals, and limitations without claiming a fixed gain."
        )
    return {"markdown": f"# {title}\n\n" + "\n\n".join(sections) + "\n"}


class KnowledgeBaseBuilderTests(unittest.TestCase):
    def test_overlong_markdown_is_truncated_without_a_repair_attempt(self):
        progress = []
        builder = KnowledgeBaseBuilder(
            ScriptedLLMClient([]),
            KNOWLEDGE_ROOT,
            minimum_card_characters=100,
            maximum_card_characters=700,
            progress=progress.append,
        )
        markdown = builder._validate_markdown(
            card(),
            {
                "id": "training.gradient_conflict",
                "title": "Multitask gradient-conflict management",
            },
        )
        self.assertEqual(len(markdown), 700)
        self.assertTrue(markdown.startswith("# Multitask gradient-conflict management"))
        self.assertIn("Truncated training.gradient_conflict", progress[0])

    def test_web_citations_are_required_and_appended_to_card(self):
        builder = KnowledgeBaseBuilder(
            ScriptedLLMClient([]),
            KNOWLEDGE_ROOT,
            minimum_card_characters=100,
            require_web_citations=True,
        )
        response = card()
        with self.assertRaisesRegex(ValueError, "at least one URL citation"):
            builder._validate_markdown(response, {"title": "Multitask gradient-conflict management"})
        response["_web_search"] = {"citations": [{
            "title": "Primary research paper",
            "url": "https://example.org/primary-paper",
        }]}
        markdown = builder._validate_markdown(
            response, {"title": "Multitask gradient-conflict management"}
        )
        self.assertIn("### Audited web sources", markdown)
        self.assertIn("<https://example.org/primary-paper>", markdown)

    def test_extend_stages_generates_manifest_and_preserves_existing_cards(self):
        llm = ScriptedLLMClient([plan(), card()])
        initial_card_count = len(KnowledgeCatalog(KNOWLEDGE_ROOT).entries)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "knowledge"
            shutil.copytree(KNOWLEDGE_ROOT, output)
            builder = KnowledgeBaseBuilder(
                llm, output, minimum_cards_per_category=1,
                maximum_cards_per_category=1, minimum_card_characters=100,
            )
            result = builder.build(categories=["training"])
            catalog = KnowledgeCatalog(output)
            self.assertEqual(result["generated_document_ids"], ["training.gradient_conflict"])
            self.assertEqual(result["total_card_count"], initial_card_count + 1)
            self.assertIn("task.kuairand_ranking", catalog.by_id)
            self.assertIn("training.gradient_conflict", catalog.by_id)
            generated = catalog.by_id["training.gradient_conflict"]
            self.assertNotIn("compute_cost", generated)
            self.assertNotIn("leakage_risk", generated)
            self.assertNotIn("evidence_level", generated)
            self.assertTrue(Path(result["backup_dir"]).is_dir())
            self.assertEqual(len(llm.calls), 2)

    def test_curator_can_choose_card_count_within_configured_range(self):
        planned = plan()
        planned["cards"].extend(plan(
            slug="optimizer_interference",
            title="Optimizer interference across ranking tasks",
        )["cards"])
        llm = ScriptedLLMClient([
            planned,
            card(),
            card(title="Optimizer interference across ranking tasks"),
        ])
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "knowledge"
            shutil.copytree(KNOWLEDGE_ROOT, output)
            builder = KnowledgeBaseBuilder(
                llm, output, minimum_cards_per_category=1,
                maximum_cards_per_category=3, minimum_card_characters=100,
            )
            result = builder.build(categories=["training"])
            self.assertEqual(result["generated_card_count"], 2)
            self.assertEqual(llm.calls[0]["payload"]["minimum_card_count"], 1)
            self.assertEqual(llm.calls[0]["payload"]["maximum_card_count"], 3)
            self.assertIn("Do not pad", llm.calls[0]["payload"]["selection_rule"])

    def test_invalid_card_is_dropped_after_attempts_and_catalog_remains_valid(self):
        planned = plan()
        planned["cards"].extend(plan(
            slug="optimizer_interference",
            title="Optimizer interference across ranking tasks",
        )["cards"])
        progress = []
        llm = ScriptedLLMClient([
            planned,
            {"markdown": "too short"},
            {"markdown": "still too short"},
            card(title="Optimizer interference across ranking tasks"),
        ])
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "knowledge"
            shutil.copytree(KNOWLEDGE_ROOT, output)
            builder = KnowledgeBaseBuilder(
                llm, output,
                minimum_cards_per_category=1,
                maximum_cards_per_category=2,
                max_response_attempts=2,
                minimum_card_characters=100,
                progress=progress.append,
            )
            result = builder.build(categories=["training"])
            catalog = KnowledgeCatalog(output)
            self.assertEqual(result["generated_document_ids"], [
                "training.optimizer_interference",
            ])
            self.assertEqual(result["generated_card_count"], 1)
            self.assertEqual(result["dropped_card_count"], 1)
            self.assertEqual(result["dropped_cards"][0]["id"], "training.gradient_conflict")
            self.assertIn("markdown length", result["dropped_cards"][0]["reason"])
            self.assertNotIn("training.gradient_conflict", catalog.by_id)
            self.assertIn("training.optimizer_interference", catalog.by_id)
            self.assertTrue(any(
                message.startswith("Dropped training.gradient_conflict")
                for message in progress
            ))

    def test_card_request_errors_are_retried_then_dropped(self):
        llm = ScriptedLLMClient([plan()])
        initial_card_count = len(KnowledgeCatalog(KNOWLEDGE_ROOT).entries)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "knowledge"
            shutil.copytree(KNOWLEDGE_ROOT, output)
            builder = KnowledgeBaseBuilder(
                llm, output,
                minimum_cards_per_category=1,
                maximum_cards_per_category=1,
                max_response_attempts=2,
            )
            result = builder.build(categories=["training"])
            self.assertEqual(result["generated_card_count"], 0)
            self.assertEqual(result["dropped_card_count"], 1)
            self.assertIn("LLM request failed", result["dropped_cards"][0]["reason"])
            self.assertEqual(len(KnowledgeCatalog(output).entries), initial_card_count)
            self.assertEqual(len(llm.calls), 3)

    def test_invalid_plan_and_markdown_are_returned_for_llm_repair(self):
        invalid_plan = plan()
        invalid_plan["cards"][0]["slug"] = "Bad Slug"
        llm = ScriptedLLMClient([
            invalid_plan,
            plan(slug="optimizer_interference", title="Optimizer interference across ranking tasks"),
            {"markdown": "too short"},
            card(title="Optimizer interference across ranking tasks"),
        ])
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "knowledge"
            shutil.copytree(KNOWLEDGE_ROOT, output)
            builder = KnowledgeBaseBuilder(
                llm, output, minimum_cards_per_category=1,
                maximum_cards_per_category=1, minimum_card_characters=100,
            )
            builder.build(categories=["training"])
            self.assertIn("slug must match", llm.calls[1]["payload"]["validation_feedback"])
            self.assertIn("markdown length", llm.calls[3]["payload"]["validation_feedback"])
            self.assertIn("training.optimizer_interference", KnowledgeCatalog(output).by_id)

    def test_replace_requires_confirmation_before_any_llm_call(self):
        llm = ScriptedLLMClient([])
        with tempfile.TemporaryDirectory() as directory:
            builder = KnowledgeBaseBuilder(llm, Path(directory) / "knowledge")
            with self.assertRaisesRegex(ValueError, "explicit confirmation"):
                builder.build(categories=["training"], mode="replace")
            self.assertEqual(llm.calls, [])

    def test_task_and_dataset_categories_are_not_llm_generated(self):
        self.assertEqual(SYSTEM_OWNED_CATEGORIES, ("task", "dataset"))
        self.assertNotIn("task", GENERATED_CATEGORIES)
        self.assertNotIn("dataset", GENERATED_CATEGORIES)
        llm = ScriptedLLMClient([])
        with self.assertRaisesRegex(ValueError, "system-owned generation categories"):
            KnowledgeBaseBuilder(llm, KNOWLEDGE_ROOT).build(categories=["dataset"])
        self.assertEqual(llm.calls, [])

    def test_confirmed_replace_retains_only_fixed_cards_and_new_research(self):
        llm = ScriptedLLMClient([plan(), card()])
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "knowledge"
            shutil.copytree(KNOWLEDGE_ROOT, output)
            builder = KnowledgeBaseBuilder(
                llm, output, minimum_cards_per_category=1,
                maximum_cards_per_category=1, minimum_card_characters=100,
            )
            result = builder.build(
                categories=["training"], mode="replace", confirm_replace=True
            )
            catalog = KnowledgeCatalog(output)
            self.assertEqual(result["total_card_count"], 12)
            self.assertIn("training.gradient_conflict", catalog.by_id)
            self.assertIn("task.leakage_policy", catalog.by_id)
            self.assertIn("dataset.inventory_and_splits", catalog.by_id)
            self.assertNotIn("training.regularization", catalog.by_id)
            self.assertEqual(len(catalog.fixed_documents()), 5)

    def test_failed_generation_leaves_live_knowledge_base_untouched(self):
        llm = ScriptedLLMClient([{"cards": []}])
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "knowledge"
            shutil.copytree(KNOWLEDGE_ROOT, output)
            before = KnowledgeCatalog(output).catalog_hash
            builder = KnowledgeBaseBuilder(
                llm, output, minimum_cards_per_category=1,
                maximum_cards_per_category=1, max_response_attempts=1,
            )
            with self.assertRaisesRegex(Exception, "knowledge plan remained invalid"):
                builder.build(categories=["training"])
            self.assertEqual(KnowledgeCatalog(output).catalog_hash, before)
            self.assertEqual(list(Path(directory).glob(".knowledge-staging-*")), [])


if __name__ == "__main__":
    unittest.main()
