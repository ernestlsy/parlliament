from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .llm import LLMClient, LLMError


JSON_ONLY = "Return only one valid JSON object. Do not use Markdown fences."
CATALOG_REQUIRED_FIELDS = {
    "id", "path", "title", "category", "tags", "summary", "use_when", "avoid_when",
    "required_features", "metrics",
}
TOKEN_PATTERN = re.compile(r"[a-z0-9][a-z0-9_+-]*")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_text_sha256(data: bytes) -> str:
    """Hash UTF-8 text after normalizing line endings for Windows/Linux portability."""
    canonical = data.decode("utf-8").replace("\r\n", "\n").replace("\r", "\n")
    return _sha256(canonical.encode("utf-8"))


def _tokens(value: Any) -> List[str]:
    return TOKEN_PATTERN.findall(str(value).lower().replace("-", "_"))


def _json_write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


@dataclass(frozen=True)
class ResearchRequest:
    query: str
    purpose: str
    categories: Tuple[str, ...]
    preferred_tags: Tuple[str, ...]
    metrics_of_interest: Tuple[str, ...]
    max_documents: int

    @classmethod
    def from_dict(
        cls,
        value: Mapping[str, Any],
        *,
        allowed_categories: Iterable[str],
        max_documents: int,
    ) -> "ResearchRequest":
        query = str(value.get("query", "")).strip()
        purpose = str(value.get("purpose", "")).strip()
        if not query or not purpose:
            raise ValueError("research request requires non-empty query and purpose")

        def strings(name: str) -> Tuple[str, ...]:
            raw = value.get(name, [])
            if not isinstance(raw, list) or not all(isinstance(item, str) for item in raw):
                raise ValueError(f"research request {name} must be a string list")
            return tuple(dict.fromkeys(item.strip() for item in raw if item.strip()))

        categories = strings("categories")
        unknown = set(categories) - set(allowed_categories)
        if unknown:
            raise ValueError(f"research request uses unknown categories: {sorted(unknown)}")
        requested_max = int(value.get("max_documents", max_documents))
        if not 1 <= requested_max <= max_documents:
            raise ValueError(f"research request max_documents must be between 1 and {max_documents}")
        return cls(
            query=query,
            purpose=purpose,
            categories=categories,
            preferred_tags=strings("preferred_tags"),
            metrics_of_interest=strings("metrics_of_interest"),
            max_documents=requested_max,
        )


@dataclass(frozen=True)
class RetrievalSettings:
    deterministic_candidates: int = 10
    assisted_candidates: int = 10
    alternative_queries: int = 5
    results_per_alternative: int = 2
    max_final_documents: int = 8
    character_budget: int = 40_000

    def validate(self) -> None:
        for name in (
            "deterministic_candidates", "assisted_candidates", "alternative_queries",
            "results_per_alternative", "max_final_documents", "character_budget",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")


class KnowledgeCatalog:
    """Validated catalog and system-owned Markdown fetcher."""

    def __init__(self, root: Path):
        self.root = Path(root).resolve()
        self.catalog_path = self.root / "catalog.jsonl"
        self.manifest_path = self.root / "manifest.json"
        if not self.catalog_path.is_file() or not self.manifest_path.is_file():
            raise FileNotFoundError("knowledge catalog requires catalog.jsonl and manifest.json")
        self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        self.entries = self._load_entries()
        self.by_id = {item["id"]: item for item in self.entries}
        self.allowed_categories = tuple(self.manifest.get("allowed_categories", []))
        self.catalog_hash = canonical_text_sha256(self.catalog_path.read_bytes())
        self._validate_manifest()

    def _resolve(self, relative: str) -> Path:
        pure = PurePosixPath(relative)
        if pure.is_absolute() or ".." in pure.parts or pure.suffix.lower() != ".md":
            raise ValueError(f"unsafe knowledge-card path: {relative!r}")
        resolved = (self.root / Path(*pure.parts)).resolve()
        try:
            resolved.relative_to(self.root)
        except ValueError as exc:
            raise ValueError(f"knowledge-card path escapes root: {relative!r}") from exc
        return resolved

    def _load_entries(self) -> List[Dict[str, Any]]:
        entries = []
        identifiers = set()
        with self.catalog_path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid catalog JSON on line {line_number}: {exc}") from exc
                missing = CATALOG_REQUIRED_FIELDS - set(item)
                if missing:
                    raise ValueError(
                        f"catalog entry on line {line_number} is missing {sorted(missing)}"
                    )
                identifier = str(item["id"])
                if not identifier or identifier in identifiers:
                    raise ValueError(f"duplicate or empty catalog ID: {identifier!r}")
                identifiers.add(identifier)
                for field in (
                    "tags", "use_when", "avoid_when", "required_features", "metrics",
                ):
                    if not isinstance(item[field], list) or not all(
                        isinstance(value, str) for value in item[field]
                    ):
                        raise ValueError(f"catalog {identifier} field {field} must be a string list")
                path = self._resolve(str(item["path"]))
                if not path.is_file():
                    raise FileNotFoundError(f"catalog document does not exist: {item['path']}")
                item = dict(item)
                item["id"] = identifier
                item["path"] = str(PurePosixPath(str(item["path"])))
                entries.append(item)
        return entries

    def _validate_manifest(self) -> None:
        if int(self.manifest.get("schema_version", 0)) != 1:
            raise ValueError("knowledge manifest schema_version must be 1")
        if int(self.manifest.get("card_count", -1)) != len(self.entries):
            raise ValueError("knowledge manifest card_count does not match catalog")
        if self.manifest.get("catalog_sha256") != self.catalog_hash:
            raise ValueError("knowledge catalog hash does not match manifest")
        required = set(self.manifest.get("required_catalog_fields", []))
        if not CATALOG_REQUIRED_FIELDS.issubset(required):
            raise ValueError("knowledge manifest omits required catalog fields")
        categories = set(self.manifest.get("allowed_categories", []))
        used = {item["category"] for item in self.entries}
        if not used.issubset(categories):
            raise ValueError(f"catalog uses categories absent from manifest: {sorted(used-categories)}")
        hashes = self.manifest.get("document_hashes", {})
        if set(hashes) != {item["id"] for item in self.entries}:
            raise ValueError("knowledge manifest document_hashes must cover every catalog ID")
        for item in self.entries:
            actual = canonical_text_sha256(self._resolve(item["path"]).read_bytes())
            if hashes[item["id"]] != actual:
                raise ValueError(f"knowledge document hash mismatch: {item['id']}")

    @staticmethod
    def compact(item: Mapping[str, Any]) -> Dict[str, Any]:
        fields = (
            "id", "title", "category", "tags", "summary", "use_when", "avoid_when",
            "required_features", "metrics", "method_family",
        )
        return {name: item[name] for name in fields if name in item}

    def fixed_documents(self) -> List[Dict[str, str]]:
        identifiers = [item["id"] for item in self.entries if item.get("always_include")]
        documents, _ = self.fetch(identifiers, allowed_ids=identifiers, character_budget=20_000)
        return documents

    def fetch(
        self,
        identifiers: Sequence[str],
        *,
        allowed_ids: Iterable[str],
        character_budget: int,
    ) -> Tuple[List[Dict[str, str]], List[Dict[str, Any]]]:
        allowed = set(allowed_ids)
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("selected document IDs must be unique")
        if not set(identifiers).issubset(allowed):
            raise ValueError("selected document IDs must come from the merged candidate pool")
        unknown = set(identifiers) - set(self.by_id)
        if unknown:
            raise ValueError(f"unknown selected document IDs: {sorted(unknown)}")
        remaining = character_budget
        documents = []
        records = []
        for index, identifier in enumerate(identifiers):
            item = self.by_id[identifier]
            path = self._resolve(item["path"])
            raw = path.read_bytes()
            digest = canonical_text_sha256(raw)
            if digest != self.manifest["document_hashes"][identifier]:
                raise ValueError(f"knowledge document changed after catalog load: {identifier}")
            text = raw.decode("utf-8")
            documents_left = len(identifiers) - index
            allowance = max(0, remaining // documents_left)
            content = text[:allowance]
            remaining -= len(content)
            documents.append({
                "id": identifier,
                "title": str(item["title"]),
                "source": item["path"],
                "content": content,
                "truncated": str(len(content) < len(text)).lower(),
            })
            records.append({
                "id": identifier,
                "path": item["path"],
                "sha256": digest,
                "source_characters": len(text),
                "fetched_characters": len(content),
                "truncated": len(content) < len(text),
            })
        if sum(len(item["content"]) for item in documents) > character_budget:
            raise AssertionError("literature fetch exceeded character budget")
        return documents, records


class CatalogRetriever:
    """Deterministic metadata TF-IDF retrieval with MMR diversity."""

    _INDEX_CACHE: Dict[str, Tuple[Dict[str, float], Dict[str, Dict[str, float]]]] = {}

    def __init__(self, catalog: KnowledgeCatalog):
        self.catalog = catalog
        cached = self._INDEX_CACHE.get(catalog.catalog_hash)
        if cached is None:
            cached = self._build_index(catalog.entries)
            self._INDEX_CACHE[catalog.catalog_hash] = cached
            self.cache_status = "built"
        else:
            self.cache_status = "memory_hit"
        self.idf, self.vectors = cached

    @staticmethod
    def _entry_text(item: Mapping[str, Any]) -> str:
        values = [
            item["title"], item["category"], item["summary"], item.get("method_family", ""),
            " ".join(item["tags"]), " ".join(item["use_when"]),
            " ".join(item["avoid_when"]), " ".join(item["required_features"]),
            " ".join(item["metrics"]),
        ]
        return " ".join(str(value) for value in values)

    @classmethod
    def _build_index(
        cls, entries: Sequence[Mapping[str, Any]]
    ) -> Tuple[Dict[str, float], Dict[str, Dict[str, float]]]:
        counters = {item["id"]: Counter(_tokens(cls._entry_text(item))) for item in entries}
        document_frequency = Counter()
        for counts in counters.values():
            document_frequency.update(counts.keys())
        size = max(1, len(entries))
        idf = {
            token: math.log((1.0 + size) / (1.0 + frequency)) + 1.0
            for token, frequency in document_frequency.items()
        }
        vectors = {}
        for identifier, counts in counters.items():
            weighted = {token: count * idf[token] for token, count in counts.items()}
            norm = math.sqrt(sum(value * value for value in weighted.values())) or 1.0
            vectors[identifier] = {token: value / norm for token, value in weighted.items()}
        return idf, vectors

    def _query_vector(self, text: str) -> Dict[str, float]:
        counts = Counter(_tokens(text))
        weighted = {
            token: count * self.idf[token] for token, count in counts.items() if token in self.idf
        }
        norm = math.sqrt(sum(value * value for value in weighted.values())) or 1.0
        return {token: value / norm for token, value in weighted.items()}

    @staticmethod
    def _cosine(left: Mapping[str, float], right: Mapping[str, float]) -> float:
        if len(left) > len(right):
            left, right = right, left
        return sum(value * right.get(token, 0.0) for token, value in left.items())

    def _eligible(
        self, item: Mapping[str, Any], request: ResearchRequest, context: Mapping[str, Any]
    ) -> bool:
        if request.categories and item["category"] not in request.categories:
            return False
        tested = str(context.get("tested_hypotheses", "")).lower().replace("-", "_")
        family = str(item.get("method_family", "")).lower().replace("-", "_")
        if family and family.replace("_", " ") in tested.replace("_", " "):
            return False
        return True

    def search(
        self,
        request: ResearchRequest,
        context: Mapping[str, Any],
        *,
        query: Optional[str] = None,
        limit: int = 10,
        excluded_ids: Iterable[str] = (),
    ) -> List[Dict[str, Any]]:
        excluded = set(excluded_ids)
        enriched = " ".join([
            query or request.query,
            request.purpose,
            " ".join(request.categories),
            " ".join(request.preferred_tags),
            " ".join(request.metrics_of_interest),
            str(context.get("metric_weaknesses", "")),
            str(context.get("current_architecture", "")),
        ])
        query_vector = self._query_vector(enriched)
        scored = []
        preferred_tags = set(request.preferred_tags)
        metrics = set(request.metrics_of_interest)
        for item in self.catalog.entries:
            if item["id"] in excluded or not self._eligible(item, request, context):
                continue
            score = self._cosine(query_vector, self.vectors[item["id"]])
            score += 0.08 * len(preferred_tags.intersection(item["tags"]))
            score += 0.04 * len(metrics.intersection(item["metrics"]))
            if item["category"] in request.categories:
                score += 0.08
            scored.append((score, item))
        scored.sort(key=lambda pair: (-pair[0], pair[1]["id"]))

        selected: List[Tuple[float, Mapping[str, Any]]] = []
        remaining = list(scored)
        while remaining and len(selected) < limit:
            best_index = 0
            best_value = float("-inf")
            for index, (relevance, item) in enumerate(remaining):
                redundancy = max(
                    (
                        self._cosine(
                            self.vectors[item["id"]], self.vectors[chosen["id"]]
                        )
                        for _, chosen in selected
                    ),
                    default=0.0,
                )
                mmr = 0.75 * relevance - 0.25 * redundancy
                tie_key = (mmr, relevance, item["id"])
                best = remaining[best_index]
                best_tie = (best_value, best[0], best[1]["id"])
                if tie_key > best_tie:
                    best_index, best_value = index, mmr
            selected.append(remaining.pop(best_index))
        return [
            {**self.catalog.compact(item), "retrieval_score": round(score, 8)}
            for score, item in selected
        ]


class Librarian:
    """Hybrid literature retrieval and guarded reading-list selection."""

    def __init__(self, llm: LLMClient, knowledge_root: Path, settings: RetrievalSettings):
        settings.validate()
        self.llm = llm
        self.settings = settings
        self.catalog = KnowledgeCatalog(knowledge_root)
        self.retriever = CatalogRetriever(self.catalog)

    def _expand_queries(
        self, request: ResearchRequest, context: Mapping[str, Any]
    ) -> Tuple[List[str], Optional[str]]:
        system = (
            "Generate alternative literature-search queries for a recommender-system research "
            "request. Use related terminology and useful conceptual connections without changing "
            "the task, leakage boundary, or requested metrics. " + JSON_ONLY
        )
        payload = {
            "research_request": asdict(request),
            "experiment_context": dict(context),
            "available_categories": list(self.catalog.allowed_categories),
            "requested_query_count": self.settings.alternative_queries,
            "validation_feedback": None,
            "response_schema": {"alternative_queries": ["string"]},
        }
        last_error = ""
        try:
            for attempt in range(1, 4):
                payload["response_attempt"] = attempt
                payload["validation_feedback"] = last_error or None
                result = self.llm.complete_json(
                    role="librarian_query_expansion", system=system, payload=payload
                )
                try:
                    queries = result.get("alternative_queries")
                    if not isinstance(queries, list) or not queries:
                        raise ValueError("alternative_queries must be a non-empty list")
                    cleaned = list(dict.fromkeys(
                        str(item).strip() for item in queries if str(item).strip()
                    ))
                    if not 1 <= len(cleaned) <= self.settings.alternative_queries:
                        raise ValueError(
                            f"expected 1-{self.settings.alternative_queries} unique queries"
                        )
                    return cleaned, None
                except (AttributeError, TypeError, ValueError) as exc:
                    last_error = f"Response validation failed: {type(exc).__name__}: {exc}"
        except LLMError as exc:
            last_error = f"LLM query expansion failed: {exc}"
        return [], last_error or "query expansion failed after three responses"

    def _select_ids(
        self,
        request: ResearchRequest,
        candidates: Sequence[Mapping[str, Any]],
        max_documents: int,
    ) -> Tuple[List[str], Optional[str]]:
        if not candidates or max_documents <= 0:
            return [], None
        system = (
            "You are the Librarian. Select the smallest useful, diverse reading list for the "
            "research request from the supplied candidate metadata. Return only document IDs from "
            "the candidate pool. Do not return paths, scores, explanations, or invented IDs. "
            + JSON_ONLY
        )
        candidate_ids = [str(item["id"]) for item in candidates]
        minimum = min(5, max_documents, len(candidate_ids))
        payload = {
            "research_request": asdict(request),
            "candidate_documents": [
                {key: value for key, value in item.items() if key != "retrieval_score"}
                for item in candidates
            ],
            "minimum_documents": minimum,
            "maximum_documents": max_documents,
            "validation_feedback": None,
            "response_schema": {"selected_document_ids": ["document.id"]},
        }
        allowed = set(candidate_ids)
        last_error = ""
        try:
            for attempt in range(1, 4):
                payload["response_attempt"] = attempt
                payload["validation_feedback"] = last_error or None
                result = self.llm.complete_json(
                    role="librarian_selection", system=system, payload=payload
                )
                try:
                    selected = result.get("selected_document_ids")
                    if not isinstance(selected, list) or not all(
                        isinstance(item, str) for item in selected
                    ):
                        raise ValueError("selected_document_ids must be a string list")
                    if len(selected) != len(set(selected)):
                        raise ValueError("selected_document_ids must be unique")
                    if not minimum <= len(selected) <= max_documents:
                        raise ValueError(
                            f"selected_document_ids requires {minimum}-{max_documents} IDs"
                        )
                    unknown = set(selected) - allowed
                    if unknown:
                        raise ValueError(f"selected IDs are outside candidate pool: {sorted(unknown)}")
                    return selected, None
                except (AttributeError, TypeError, ValueError) as exc:
                    last_error = f"Response validation failed: {type(exc).__name__}: {exc}"
        except LLMError as exc:
            last_error = f"LLM selection failed: {exc}"
        return candidate_ids[:max_documents], last_error or "selection failed after three responses"

    @staticmethod
    def _merge_candidates(
        deterministic: Sequence[Mapping[str, Any]], assisted: Sequence[Mapping[str, Any]]
    ) -> List[Dict[str, Any]]:
        merged = []
        seen_ids = set()
        seen_families = set()
        for item in list(deterministic) + list(assisted):
            identifier = str(item["id"])
            family = str(item.get("method_family", identifier))
            if identifier in seen_ids or family in seen_families:
                continue
            seen_ids.add(identifier)
            seen_families.add(family)
            merged.append(dict(item))
        return merged

    def retrieve(
        self,
        requests: Sequence[ResearchRequest],
        *,
        context: Mapping[str, Any],
        round_number: int,
        excluded_document_ids: Iterable[str] = (),
        remaining_document_slots: Optional[int] = None,
        character_budget: Optional[int] = None,
    ) -> Dict[str, Any]:
        cache_status = self.retriever.cache_status
        excluded = set(excluded_document_ids)
        remaining_slots = (
            self.settings.max_final_documents
            if remaining_document_slots is None else remaining_document_slots
        )
        audit_requests = []
        deterministic_audit = []
        expansion_audit = []
        assisted_audit = []
        merged_audit = []
        selection_audit = []
        selected_all: List[str] = []
        merged_pool_ids = set()

        for request_index, request in enumerate(requests, 1):
            if remaining_slots <= 0:
                break
            audit_requests.append({
                "round": round_number, "request_index": request_index, **asdict(request)
            })
            deterministic = self.retriever.search(
                request,
                context,
                limit=self.settings.deterministic_candidates,
                excluded_ids=excluded,
            )
            deterministic_audit.append({
                "round": round_number,
                "request_index": request_index,
                "candidates": deterministic,
            })
            alternatives, expansion_error = self._expand_queries(request, context)
            expansion_audit.append({
                "round": round_number,
                "request_index": request_index,
                "alternative_queries": alternatives,
                "fallback_reason": expansion_error,
            })
            assisted = []
            assisted_seen = set()
            for alternative in alternatives:
                results = self.retriever.search(
                    request,
                    context,
                    query=alternative,
                    limit=self.settings.results_per_alternative,
                    excluded_ids=excluded | assisted_seen,
                )
                for item in results:
                    if len(assisted) >= self.settings.assisted_candidates:
                        break
                    assisted.append(item)
                    assisted_seen.add(item["id"])
                if len(assisted) >= self.settings.assisted_candidates:
                    break
            assisted_audit.append({
                "round": round_number,
                "request_index": request_index,
                "candidates": assisted,
            })
            merged = self._merge_candidates(deterministic, assisted)
            merged_pool_ids.update(str(item["id"]) for item in merged)
            merged_audit.append({
                "round": round_number,
                "request_index": request_index,
                "candidates": merged,
            })
            request_limit = min(request.max_documents, remaining_slots, len(merged))
            selected, selection_error = self._select_ids(request, merged, request_limit)
            selection_audit.append({
                "round": round_number,
                "request_index": request_index,
                "selected_document_ids": selected,
                "fallback_reason": selection_error,
            })
            selected_all.extend(selected)
            excluded.update(selected)
            remaining_slots -= len(selected)

        documents, fetch_records = self.catalog.fetch(
            selected_all,
            allowed_ids=merged_pool_ids,
            character_budget=(
                self.settings.character_budget if character_budget is None else character_budget
            ),
        )
        result = {
            "selected_document_ids": selected_all,
            "documents": documents,
            "audit": {
                "research_requests": audit_requests,
                "deterministic_candidates": deterministic_audit,
                "llm_expanded_queries": expansion_audit,
                "llm_retrieval_candidates": assisted_audit,
                "merged_candidates": merged_audit,
                "selected_document_ids": selection_audit,
                "fetch_records": fetch_records,
                "cache_status": cache_status,
            },
        }
        self.retriever.cache_status = "memory_hit"
        return result

    def write_audit(self, literature_dir: Path, rounds: Sequence[Mapping[str, Any]]) -> None:
        names = (
            "research_requests", "deterministic_candidates", "llm_expanded_queries",
            "llm_retrieval_candidates", "merged_candidates", "selected_document_ids",
        )
        for name in names:
            values = []
            for result in rounds:
                values.extend(result["audit"][name])
            _json_write(literature_dir / f"{name}.json", values)
        fetch_records = []
        selected = []
        cache_status = []
        for result in rounds:
            fetch_records.extend(result["audit"]["fetch_records"])
            selected.extend(result["selected_document_ids"])
            cache_status.append(result["audit"]["cache_status"])
        _json_write(literature_dir / "retrieval_manifest.json", {
            "schema_version": 1,
            "retrieval_rounds": len(rounds),
            "selected_document_ids": selected,
            "documents": fetch_records,
            "total_fetched_characters": sum(
                int(item["fetched_characters"]) for item in fetch_records
            ),
            "character_budget": self.settings.character_budget,
            "cache_status": cache_status,
            "catalog_sha256": self.catalog.catalog_hash,
        })
