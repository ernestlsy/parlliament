from __future__ import annotations

import argparse
import json
import re
import shlex
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .librarian import CATALOG_REQUIRED_FIELDS, KnowledgeCatalog, canonical_text_sha256
from .llm import AuditedLLMClient, CommandLLMClient, LLMClient, LLMError, OpenAICompatibleClient


CATEGORY_LAYOUT: Dict[str, Tuple[str, str]] = {
    "task": ("00_task", "task"),
    "dataset": ("05_dataset", "dataset"),
    "features": ("10_features", "features"),
    "architectures": ("20_architectures", "architecture"),
    "objectives": ("30_objectives", "objective"),
    "training": ("40_training", "training"),
    "evaluation": ("50_evaluation", "evaluation"),
    "bias_and_robustness": ("60_bias_and_robustness", "robustness"),
    "efficiency": ("70_efficiency", "efficiency"),
    "experiment_strategy": ("80_experiment_strategy", "experiment"),
    "papers": ("90_papers", "paper"),
}
SYSTEM_OWNED_CATEGORIES = ("task", "dataset")
GENERATED_CATEGORIES = tuple(
    category for category in CATEGORY_LAYOUT if category not in SYSTEM_OWNED_CATEGORIES
)
LIST_FIELDS = ("tags", "use_when", "avoid_when", "required_features", "metrics")
SLUG_PATTERN = re.compile(r"^[a-z][a-z0-9_]{2,63}$")
REQUIRED_HEADINGS = (
    "Summary and mechanism",
    "When to use / avoid",
    "Requirements and implementation",
    "Starting configuration and expected effects",
    "Diagnostics and risks",
    "Cheapest check and clean experiment",
    "Related cards and sources",
)
DEFAULT_README = """# Research knowledge base

Markdown cards are the source of truth. `catalog.jsonl` is the discovery interface used by the
Librarian, and document IDs--not paths--are the public retrieval identifiers. Generated cards must
be reviewed before relying on their research claims or source citations.
"""


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def _write_text(path: Path, value: str) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(value)


def _normalized_words(value: str) -> set:
    return set(re.findall(r"[a-z0-9]+", value.lower()))


def _similarity(left: str, right: str) -> float:
    first, second = _normalized_words(left), _normalized_words(right)
    if not first or not second:
        return 0.0
    return len(first & second) / len(first | second)


class KnowledgeBaseBuilder:
    """One-time, staged LLM generator for ParLLiaMent's validated literature catalog."""

    def __init__(
        self,
        llm: LLMClient,
        output_dir: Path,
        *,
        minimum_cards_per_category: int = 4,
        maximum_cards_per_category: int = 8,
        max_response_attempts: int = 3,
        minimum_card_characters: int = 900,
        maximum_card_characters: int = 10_000,
        guidance: str = "",
        require_web_citations: bool = False,
        progress: Optional[Callable[[str], None]] = None,
    ):
        self.llm = llm
        self.output_dir = Path(output_dir).resolve()
        self.minimum_cards_per_category = minimum_cards_per_category
        self.maximum_cards_per_category = maximum_cards_per_category
        self.max_response_attempts = max_response_attempts
        self.minimum_card_characters = minimum_card_characters
        self.maximum_card_characters = maximum_card_characters
        self.guidance = guidance.strip()
        self.require_web_citations = require_web_citations
        self.progress = progress or (lambda _message: None)
        if minimum_cards_per_category <= 0:
            raise ValueError("minimum_cards_per_category must be positive")
        if maximum_cards_per_category < minimum_cards_per_category:
            raise ValueError(
                "maximum_cards_per_category must be greater than or equal to the minimum"
            )
        if max_response_attempts <= 0:
            raise ValueError("max_response_attempts must be positive")
        if minimum_card_characters <= 0 or maximum_card_characters < minimum_card_characters:
            raise ValueError("card character limits are invalid")

    @staticmethod
    def _entry_summary(entries: Iterable[Mapping[str, Any]]) -> List[Dict[str, Any]]:
        fields = ("id", "category", "title", "summary", "method_family")
        return [{field: item[field] for field in fields if field in item} for item in entries]

    def _validate_plan(
        self,
        response: Mapping[str, Any],
        category: str,
        existing: Sequence[Mapping[str, Any]],
    ) -> List[Dict[str, Any]]:
        raw = response.get("cards")
        if not isinstance(raw, list) or not (
            self.minimum_cards_per_category <= len(raw) <= self.maximum_cards_per_category
        ):
            raise ValueError(
                f"cards must contain {self.minimum_cards_per_category}-"
                f"{self.maximum_cards_per_category} objects for {category}"
            )
        directory, namespace = CATEGORY_LAYOUT[category]
        used_ids = {str(item["id"]) for item in existing}
        used_paths = {str(item["path"]) for item in existing}
        comparisons = [f"{item.get('title', '')} {item.get('summary', '')}" for item in existing]
        planned: List[Dict[str, Any]] = []
        for index, item in enumerate(raw, 1):
            if not isinstance(item, dict):
                raise ValueError(f"card {index} must be an object")
            slug = item.get("slug")
            if not isinstance(slug, str) or not SLUG_PATTERN.fullmatch(slug):
                raise ValueError(f"card {index} slug must match {SLUG_PATTERN.pattern}")
            identifier = f"{namespace}.{slug}"
            path = f"{directory}/{slug}.md"
            if identifier in used_ids or identifier in {card["id"] for card in planned}:
                raise ValueError(f"duplicate card ID: {identifier}")
            if path in used_paths or path in {card["path"] for card in planned}:
                raise ValueError(f"duplicate card path: {path}")
            title = item.get("title")
            summary = item.get("summary")
            if not isinstance(title, str) or not 5 <= len(title.strip()) <= 160:
                raise ValueError(f"card {index} requires a descriptive title")
            if not isinstance(summary, str) or not 30 <= len(summary.strip()) <= 600:
                raise ValueError(f"card {index} requires a substantive summary")
            for field in LIST_FIELDS:
                values = item.get(field)
                if not isinstance(values, list) or not values or not all(
                    isinstance(value, str) and 1 <= len(value.strip()) <= 240 for value in values
                ):
                    raise ValueError(f"card {index} field {field} must be a non-empty string list")
                if len(values) > 16:
                    raise ValueError(f"card {index} field {field} cannot exceed 16 values")
            method_family = item.get("method_family")
            if not isinstance(method_family, str) or not SLUG_PATTERN.fullmatch(method_family):
                raise ValueError(f"card {index} method_family must be a lowercase underscore slug")
            candidate_text = f"{title} {summary}"
            if any(_similarity(candidate_text, other) >= 0.82 for other in comparisons):
                raise ValueError(f"card {index} is too similar to an existing or planned card")
            comparisons.append(candidate_text)
            planned.append({
                "id": identifier,
                "path": path,
                "title": title.strip(),
                "category": category,
                "tags": list(dict.fromkeys(value.strip() for value in item["tags"])),
                "summary": summary.strip(),
                "use_when": [value.strip() for value in item["use_when"]],
                "avoid_when": [value.strip() for value in item["avoid_when"]],
                "required_features": [value.strip() for value in item["required_features"]],
                "metrics": [value.strip() for value in item["metrics"]],
                "method_family": method_family,
            })
        return planned

    def _plan_category(
        self,
        category: str,
        existing: Sequence[Mapping[str, Any]],
    ) -> List[Dict[str, Any]]:
        feedback = ""
        for _ in range(self.max_response_attempts):
            response = self.llm.complete_json(
                role="knowledge_curator_plan",
                system=(
                    "You are a senior recommender-systems research curator. Plan practical, "
                    "non-duplicative literature cards for an autonomous experiment judge. Cover "
                    "architecture, feature, objective, training, evaluation, robustness, efficiency, "
                    "and experimental-decision details as relevant. Prefer established methods and "
                    "actionable configurations over broad surveys. "
                    + (
                        "Use the required web search to ground topic selection in primary research. "
                        if self.require_web_citations else ""
                    )
                    + "Return one JSON object only."
                ),
                payload={
                    "task": "plan knowledge cards",
                    "category": category,
                    "minimum_card_count": self.minimum_cards_per_category,
                    "maximum_card_count": self.maximum_cards_per_category,
                    "selection_rule": (
                        "Choose the number of cards within the allowed range based on how many "
                        "distinct, high-value topics this category genuinely supports. Do not pad "
                        "the list with weak, broad, or overlapping topics merely to reach the maximum."
                    ),
                    "existing_cards": self._entry_summary(existing),
                    "project_context": (
                        "KuaiRand-style implicit-feedback ranking; scarce official experiments; "
                        "primary metric combines GAUC and nDCG@5; Windows/Linux compatible code"
                    ),
                    "curator_guidance": self.guidance,
                    "response_schema": {
                        "cards": [{
                            "slug": "lowercase_underscore_slug",
                            "title": "string",
                            "tags": ["string"],
                            "summary": "at least 30 characters",
                            "use_when": ["string"],
                            "avoid_when": ["string"],
                            "required_features": ["string; use none when applicable"],
                            "metrics": ["GAUC", "nDCG@5"],
                            "method_family": "lowercase_underscore_slug",
                        }],
                    },
                    "validation_feedback": feedback,
                },
            )
            try:
                return self._validate_plan(response, category, existing)
            except (TypeError, ValueError) as exc:
                feedback = f"Response validation failed: {type(exc).__name__}: {exc}"
        raise LLMError(f"knowledge plan remained invalid for {category}; {feedback}")

    def _validate_markdown(self, response: Mapping[str, Any], entry: Mapping[str, Any]) -> str:
        markdown = response.get("markdown")
        if not isinstance(markdown, str):
            raise ValueError("markdown must be a string")
        markdown = markdown.replace("\r\n", "\n").replace("\r", "\n").strip() + "\n"
        if self.require_web_citations:
            metadata = response.get("_web_search", {})
            citations = metadata.get("citations", []) if isinstance(metadata, dict) else []
            if not isinstance(citations, list) or not citations:
                raise ValueError("web search must provide at least one URL citation for each card")
            source_lines = []
            seen_urls = set()
            for citation in citations:
                if not isinstance(citation, dict):
                    continue
                url = str(citation.get("url", "")).strip()
                title = str(citation.get("title", "")).strip().replace("\n", " ")
                if not url.startswith(("https://", "http://")) or url in seen_urls:
                    continue
                seen_urls.add(url)
                source_lines.append(f"- {title or url}: <{url}>")
            if not source_lines:
                raise ValueError("web search returned no valid HTTP URL citations")
            markdown += "\n### Audited web sources\n\n" + "\n".join(source_lines) + "\n"
        if len(markdown) < self.minimum_card_characters:
            raise ValueError(
                f"markdown length must be at least {self.minimum_card_characters} "
                f"characters; got {len(markdown)}"
            )
        lines = markdown.splitlines()
        if not lines or lines[0].strip() != f"# {entry['title']}":
            raise ValueError("markdown must start with the exact catalog title as an H1")
        positions = []
        for heading in REQUIRED_HEADINGS:
            marker = f"## {heading}"
            try:
                positions.append(lines.index(marker))
            except ValueError as exc:
                raise ValueError(f"markdown is missing exact heading: {marker}") from exc
        if positions != sorted(positions) or len(set(positions)) != len(positions):
            raise ValueError("required headings must appear once in the prescribed order")
        if len(markdown) > self.maximum_card_characters:
            original_length = len(markdown)
            markdown = markdown[:self.maximum_card_characters]
            self.progress(
                f"Truncated {entry.get('id', entry['title'])} from {original_length} "
                f"to {len(markdown)} characters."
            )
        return markdown

    def _generate_card(
        self, entry: Mapping[str, Any], available_ids: Sequence[str]
    ) -> Tuple[Optional[str], Optional[str]]:
        feedback = ""
        for _ in range(self.max_response_attempts):
            try:
                response = self.llm.complete_json(
                    role="knowledge_card_writer",
                    system=(
                        "You are a meticulous recommender-systems research writer. Produce a standalone "
                        "practical card grounded in established research. Never invent a paper, author, "
                        "DOI, URL, result, or dataset fact. Cite primary sources only when confident and "
                        "state clearly when advice is empirical rather than a sourced finding. "
                        + (
                            "Use the required web search and ground the card in the sources it returns. "
                            if self.require_web_citations else ""
                        )
                        + "Return one JSON object only; Markdown is a JSON string value."
                    ),
                    payload={
                        "task": "write one knowledge card",
                        "catalog_metadata": dict(entry),
                        "required_first_line": f"# {entry['title']}",
                        "required_h2_headings_in_order": list(REQUIRED_HEADINGS),
                        "content_requirements": [
                            "explain the mechanism and assumptions",
                            "give implementable steps, defaults, and tuning ranges",
                            "discuss likely GAUC and nDCG@5 effects without fabricating magnitudes",
                            "cover data leakage, compute, failure modes, and diagnostic signatures",
                            "give a cheap train-only check and a clean single-variable experiment",
                            "list only relevant IDs from available_related_card_ids",
                            "include primary-source links or identifiers only when confidently known",
                        ],
                        "available_related_card_ids": list(available_ids),
                        "character_limits": {
                            "minimum": self.minimum_card_characters,
                            "maximum": self.maximum_card_characters,
                        },
                        "curator_guidance": self.guidance,
                        "response_schema": {"markdown": "string"},
                        "validation_feedback": feedback,
                    },
                )
            except LLMError as exc:
                feedback = f"LLM request failed: {exc}"
                continue
            try:
                return self._validate_markdown(response, entry), None
            except (TypeError, ValueError) as exc:
                feedback = f"Response validation failed: {type(exc).__name__}: {exc}"
        return None, feedback or "card generation exhausted without a valid response"

    @staticmethod
    def _write_catalog(root: Path, entries: Sequence[Mapping[str, Any]]) -> None:
        ordered = sorted((dict(item) for item in entries), key=lambda item: item["id"])
        catalog_text = "".join(
            json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n"
            for item in ordered
        )
        catalog_path = root / "catalog.jsonl"
        _write_text(catalog_path, catalog_text)
        manifest = {
            "schema_version": 1,
            "card_count": len(ordered),
            "catalog_sha256": canonical_text_sha256(catalog_path.read_bytes()),
            "retrieval_index_version": "tfidf_mmr_v1",
            "allowed_categories": list(CATEGORY_LAYOUT),
            "required_catalog_fields": sorted(CATALOG_REQUIRED_FIELDS),
            "document_hashes": {
                item["id"]: canonical_text_sha256((root / item["path"]).read_bytes())
                for item in ordered
            },
        }
        _write_json(root / "manifest.json", manifest)

    @staticmethod
    def _backup_path(output_dir: Path, stamp: str) -> Path:
        candidate = output_dir.parent / f"{output_dir.name}.backup-{stamp}"
        suffix = 1
        while candidate.exists():
            candidate = output_dir.parent / f"{output_dir.name}.backup-{stamp}-{suffix}"
            suffix += 1
        return candidate

    def build(
        self,
        *,
        categories: Sequence[str],
        mode: str = "extend",
        confirm_replace: bool = False,
    ) -> Dict[str, Any]:
        if mode not in {"extend", "replace"}:
            raise ValueError("mode must be extend or replace")
        categories = tuple(dict.fromkeys(categories))
        if not categories:
            raise ValueError("at least one category is required")
        unknown = set(categories) - set(GENERATED_CATEGORIES)
        if unknown:
            raise ValueError(
                "unknown or system-owned generation categories: "
                f"{sorted(unknown)}; generated categories are {list(GENERATED_CATEGORIES)}"
            )
        if mode == "replace" and not confirm_replace:
            raise ValueError("replace mode requires explicit confirmation")

        if not self.output_dir.is_dir():
            raise FileNotFoundError(f"{mode} mode requires an existing knowledge directory")
        live_catalog = KnowledgeCatalog(self.output_dir)
        if mode == "extend":
            existing_entries = [dict(item) for item in live_catalog.entries]
        else:
            existing_entries = [
                dict(item) for item in live_catalog.entries
                if item.get("system_owned") or item.get("always_include")
            ]
            if not existing_entries:
                raise ValueError("replace mode requires at least one system-owned card")

        self.output_dir.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=".knowledge-staging-", dir=str(self.output_dir.parent)))
        generated_entries: List[Dict[str, Any]] = []
        dropped_cards: List[Dict[str, str]] = []
        backup: Optional[Path] = None
        try:
            if mode == "extend":
                shutil.copytree(self.output_dir, staging, dirs_exist_ok=True)
            else:
                readme = self.output_dir / "README.md"
                _write_text(
                    staging / "README.md",
                    readme.read_text(encoding="utf-8") if readme.is_file() else DEFAULT_README,
                )
                for entry in existing_entries:
                    destination = staging / entry["path"]
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(self.output_dir / entry["path"], destination)

            all_entries = list(existing_entries)
            for category in categories:
                self.progress(
                    f"Planning {self.minimum_cards_per_category}-"
                    f"{self.maximum_cards_per_category} card(s) for {category}..."
                )
                planned = self._plan_category(category, all_entries)
                for index, entry in enumerate(planned, 1):
                    self.progress(
                        f"Writing {category} card {index}/{len(planned)}: {entry['id']}..."
                    )
                    available_ids = [item["id"] for item in all_entries]
                    markdown, failure = self._generate_card(entry, available_ids)
                    if markdown is None:
                        reason = failure or "card generation exhausted"
                        dropped_cards.append({"id": str(entry["id"]), "reason": reason})
                        self.progress(f"Dropped {entry['id']}: {reason}")
                        continue
                    destination = staging / entry["path"]
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    _write_text(destination, markdown)
                    generated_entries.append(entry)
                    all_entries.append(entry)

            self._write_catalog(staging, all_entries)
            self.progress("Validating staged catalog, paths, and hashes...")
            validated = KnowledgeCatalog(staging)
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            if self.output_dir.exists():
                backup = self._backup_path(self.output_dir, stamp)
                self.output_dir.rename(backup)
            try:
                staging.rename(self.output_dir)
            except Exception:
                if backup is not None and not self.output_dir.exists():
                    backup.rename(self.output_dir)
                raise
            self.progress(f"Installed validated knowledge base at {self.output_dir}")
            return {
                "status": "completed",
                "mode": mode,
                "output_dir": str(self.output_dir),
                "backup_dir": str(backup) if backup else None,
                "generated_card_count": len(generated_entries),
                "dropped_card_count": len(dropped_cards),
                "total_card_count": len(validated.entries),
                "minimum_cards_per_category": self.minimum_cards_per_category,
                "maximum_cards_per_category": self.maximum_cards_per_category,
                "web_search_required": self.require_web_citations,
                "generated_document_ids": [item["id"] for item in generated_entries],
                "dropped_cards": dropped_cards,
                "categories": list(categories),
                "catalog_sha256": validated.catalog_hash,
                "completed_at": datetime.now(timezone.utc).isoformat(),
            }
        finally:
            if staging.exists():
                shutil.rmtree(staging)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="parlliament-populate-knowledge",
        description="Manually generate and validate ParLLiaMent literature cards before a run",
    )
    providers = parser.add_mutually_exclusive_group(required=True)
    providers.add_argument("--llm-command", help="local command accepting JSON stdin and emitting JSON stdout")
    providers.add_argument("--model", help="capable model for an OpenAI or compatible HTTP API")
    parser.add_argument("--base-url", default="https://api.openai.com/v1")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--api-mode", choices=["auto", "responses", "chat"], default="auto")
    parser.add_argument("--no-json-mode", action="store_true")
    parser.add_argument(
        "--disable-web-search", action="store_true",
        help="disable required hosted web search for HTTP-model generation",
    )
    parser.add_argument(
        "--web-search-context-size", choices=["low", "medium", "high"], default="medium",
    )
    parser.add_argument("--llm-timeout", type=int, default=300)
    parser.add_argument("--llm-retries", type=int, default=2)
    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).with_name("knowledge")),
        help="knowledge directory to extend or replace",
    )
    parser.add_argument("--mode", choices=["extend", "replace"], default="extend")
    parser.add_argument("--yes", action="store_true", help="required confirmation for replace mode")
    parser.add_argument(
        "--category", action="append", choices=list(GENERATED_CATEGORIES),
        help=(
            "research category to populate; repeat as needed "
            "(default: every LLM-generated category; task and dataset are system-owned)"
        ),
    )
    parser.add_argument("--minimum-cards-per-category", type=int, default=4)
    parser.add_argument("--maximum-cards-per-category", type=int, default=10)
    parser.add_argument("--response-attempts", type=int, default=2)
    parser.add_argument("--minimum-card-characters", type=int, default=900)
    parser.add_argument("--maximum-card-characters", type=int, default=10_000)
    parser.add_argument("--guidance-file", help="optional UTF-8 project guidance appended to prompts")
    parser.add_argument("--report", help="generation report path (default: beside output directory)")
    parser.add_argument("--llm-log", help="full JSONL request/response audit path")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    output_dir = Path(args.output_dir).resolve()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log_path = Path(args.llm_log).resolve() if args.llm_log else (
        output_dir.parent / f"knowledge_generation_llm_{stamp}.jsonl"
    )
    report_path = Path(args.report).resolve() if args.report else (
        output_dir.parent / f"knowledge_generation_report_{stamp}.json"
    )
    if args.llm_command:
        inner: LLMClient = CommandLLMClient(
            shlex.split(args.llm_command), timeout_seconds=args.llm_timeout
        )
        model_description = "command_adapter"
        hosted_web_search = False
    else:
        hosted_web_search = not args.disable_web_search
        inner = OpenAICompatibleClient(
            args.model,
            base_url=args.base_url,
            api_key=args.api_key,
            timeout_seconds=args.llm_timeout,
            max_retries=args.llm_retries,
            api_mode=args.api_mode,
            json_mode=not args.no_json_mode,
            web_search=hosted_web_search,
            web_search_context_size=args.web_search_context_size,
        )
        model_description = args.model
    guidance = ""
    if args.guidance_file:
        guidance = Path(args.guidance_file).read_text(encoding="utf-8")
    builder = KnowledgeBaseBuilder(
        AuditedLLMClient(inner, log_path),
        output_dir,
        minimum_cards_per_category=args.minimum_cards_per_category,
        maximum_cards_per_category=args.maximum_cards_per_category,
        max_response_attempts=args.response_attempts,
        minimum_card_characters=args.minimum_card_characters,
        maximum_card_characters=args.maximum_card_characters,
        guidance=guidance,
        require_web_citations=hosted_web_search,
        progress=lambda message: print(message, file=sys.stderr, flush=True),
    )
    try:
        result = builder.build(
            categories=args.category or list(GENERATED_CATEGORIES),
            mode=args.mode,
            confirm_replace=args.yes,
        )
    except Exception as exc:
        _write_json(report_path, {
            "status": "failed",
            "mode": args.mode,
            "output_dir": str(output_dir),
            "model": model_description,
            "llm_log": str(log_path),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "failed_at": datetime.now(timezone.utc).isoformat(),
        })
        print(f"Generation failed; report: {report_path}", file=sys.stderr, flush=True)
        raise
    result.update({
        "model": model_description,
        "llm_log": str(log_path),
        "hosted_web_search": hosted_web_search,
    })
    result["report"] = str(report_path)
    _write_json(report_path, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
