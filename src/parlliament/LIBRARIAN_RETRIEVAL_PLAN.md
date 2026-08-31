# Librarian and Literature Retrieval Plan

## Status

Implemented. This document records the design now used by ParLLiaMent's cataloged recommender-system
research knowledge base and two-pass Librarian retrieval workflow.

## Goals

- Scale the knowledge base without placing every document in every Evolution Judge prompt.
- Let the Evolution Judge request literature for a specific research question when needed.
- Combine reproducible lexical retrieval with LLM-assisted conceptual discovery.
- Return only a small, validated list of Markdown documents to fetch.
- Preserve an auditable record of each research request and retrieval result.
- Prevent arbitrary filesystem access, hallucinated paths, and unbounded context growth.

## Knowledge-base structure

Use categorized Markdown knowledge cards as the source of truth, with a machine-readable catalog
for discovery:

```text
knowledge/
├── README.md
├── catalog.jsonl
├── manifest.json
├── 00_task/
├── 05_dataset/
├── 10_features/
├── 20_architectures/
├── 30_objectives/
├── 40_training/
├── 50_evaluation/
├── 60_bias_and_robustness/
├── 70_efficiency/
├── 80_experiment_strategy/
└── 90_papers/
```

Recommended categories include:

- Fixed task facts, KuaiRand constraints, feature availability, and leakage policy
- Supplied dataset schema, split boundaries, population properties, and exposure regimes
- Feature engineering and historical or sequential signals
- Model architectures
- Pointwise, pairwise, listwise, and multitask objectives
- Training methods, negative sampling, optimization, regularization, and configuration
- Evaluation, temporal validation, segment diagnostics, and offline/online mismatch
- Exposure bias, cold start, popularity bias, and temporal drift
- Runtime, memory, embedding-table, and inference efficiency
- Experiment selection, ablation design, and convergence-budget strategy
- Individual papers or tightly related groups of papers

Each Markdown card should cover one concept or a tightly related method family. Cards should use a
consistent structure covering:

- Summary and mechanism
- When to use and when not to use
- Data and feature requirements
- Implementation recipe
- Useful starting configurations and ranges
- Expected effects on GAUC and nDCG@5
- Diagnostic signatures
- Leakage, compute, and failure risks
- Cheapest train-only preliminary check
- Clean one-change experiment design
- Related cards and source citations

## Catalog

Use `catalog.jsonl`, with one object per Markdown card. Each object should contain enough information
for retrieval without loading the full document:

```json
{"id":"architecture.factorization_machines","path":"20_architectures/factorization_machines.md","title":"Factorization Machines","category":"architectures","tags":["metadata","feature-interactions","sparse-features"],"summary":"Models pairwise interactions between sparse categorical fields.","use_when":["categorical metadata has useful screening lift"],"avoid_when":["embedding memory exceeds the budget"],"required_features":["categorical-fields"],"metrics":["GAUC","nDCG@5"],"method_family":"factorization_machine"}
```

`manifest.json` should contain the schema version, card count, catalog hash, retrieval-index version,
allowed categories, required catalog fields, and document hashes.

Document IDs are the public retrieval interface. LLMs must never construct or return arbitrary
filesystem paths.

## Evolution Judge research request

Before generating final experiment candidates, the Evolution Judge may identify a knowledge gap and
return one or more structured research requests:

```json
{
  "research_requests": [
    {
      "query": "improve cold-item ranking using video metadata",
      "purpose": "identify the most suitable architecture or training change",
      "categories": ["architectures", "features", "cold_start"],
      "preferred_tags": ["item-metadata", "cold-item", "ranking"],
      "metrics_of_interest": ["GAUC", "nDCG@5"],
      "max_documents": 6
    }
  ]
}
```

The retrieval context should also contain available features, weak metric segments, current model
architecture, previously tested hypotheses, and the experiment timeout.

## Retrieval pipeline

For each research request:

1. Apply deterministic hard filters to remove catalog entries with incompatible task assumptions
   or materially duplicated tested methods.
2. Run deterministic TF-IDF retrieval with category, tag, feature, metric, and experiment-context
   enrichment. Apply deterministic diversity selection such as maximal marginal relevance. Retain
   up to 10 candidates.
3. Ask an LLM retrieval role to generate approximately five alternative research queries that make
   useful conceptual connections or use different terminology.
4. Run the same deterministic catalog search for each alternative query and retain approximately two
   candidates per query, producing up to 10 LLM-assisted candidates.
5. Merge both pools by document ID and remove near-duplicate method-family cards when they would
   crowd out useful diversity.
6. Give the compact metadata and summaries for the merged pool to the Librarian LLM.
7. The Librarian returns only the final document IDs.
8. A system-owned fetcher validates those IDs, resolves their catalog paths, and loads the Markdown.
9. Append the fetched documents to a second Evolution Judge call, which then performs informed
   candidate generation.

The first implementation should use this explicit two-pass JSON workflow rather than provider-native
tool calling. It preserves compatibility with OpenAI Responses, Chat Completions, compatible APIs,
command-based adapters, and scripted tests. Native tool calling may be added later without changing
the Librarian interface.

## Minimal Librarian response

The Librarian does not need to expose novelty, relevance, evidence-strength, or risk scores. Its
responsibility is only to choose a reading list from the validated candidate pool:

```json
{
  "selected_document_ids": [
    "architecture.factorization_machines",
    "features.item_metadata",
    "problem.cold_start",
    "training.negative_sampling",
    "evaluation.cold_item_segments"
  ]
}
```

TF-IDF and diversity scores may be used internally to form deterministic candidate lists, but they
do not need to appear in the Librarian output or Evolution Judge context.

## Validation and guardrails

System code must enforce that:

- Every selected ID exists in `catalog.jsonl`.
- Every selected ID came from the merged candidate pool.
- IDs are unique and do not exceed the request's `max_documents` value.
- Catalog paths are relative Markdown paths with no absolute or parent traversal components.
- Resolved paths stay inside the knowledge root.
- Optional document hashes match `manifest.json`.
- Total fetched content remains within the retrieval character budget.
- Retrieved text is reference material and cannot override system instructions, fixed evaluation,
  leakage rules, file guardrails, or user requirements.

Invalid responses should be returned to the Librarian with validation feedback for up to three
response attempts.

## Context and retrieval budgets

Initial defaults:

- 10 deterministic candidates
- Up to 10 LLM-assisted candidates
- 5 alternative LLM queries with up to 2 results per query
- 5-8 final Markdown documents
- At most 2 retrieval rounds per planning generation
- 30,000-50,000 total fetched characters
- Deduplication across requests and retrieval rounds

Small fixed-task, evaluation, and leakage cards may always be loaded outside this retrieval budget.
All other literature should be fetched only when relevant.

## Audit artifacts

Store retrieval records under the corresponding planning generation:

```text
planning/generation_<n>/literature/
├── research_requests.json
├── deterministic_candidates.json
├── llm_expanded_queries.json
├── llm_retrieval_candidates.json
├── merged_candidates.json
├── selected_document_ids.json
└── retrieval_manifest.json
```

The retrieval manifest should record selected IDs, canonical relative paths, file hashes, character
counts, cache status, and retrieval rounds. It does not need subjective document scores.

## Integration sequence

The eventual planning workflow should be:

```text
Dataset screening and experiment diagnostics
    -> initial Evolution Judge research planning
    -> deterministic and LLM-assisted retrieval
    -> Librarian final reading list
    -> system-controlled Markdown fetch
    -> informed Evolution Judge candidate generation
    -> Consultant candidate comparison
    -> final hypothesis selection
    -> one counted experiment
```

Retrieved document IDs should be retained with the resulting hypothesis so later audits can trace
which literature informed each experiment. Literature retrieval itself must never create an attempt,
consume an experiment ID, or affect convergence.

## Implemented components

- Catalog schema and validator
- Knowledge-card path and hash validator
- TF-IDF index builder and in-process cache
- Deterministic filters and MMR diversity selection
- LLM query-expansion role
- Librarian selection role with deterministic fallback
- System-owned Markdown fetcher
- Two-pass Evolution Judge research interface
- Retrieval budgets and retry handling
- Planning audit artifacts
- Tests for catalog integrity, path traversal, hallucinated IDs, deduplication, deterministic results,
  context budgets, caching, provider compatibility, and experiment-count isolation

## One-time population utility

`parlliament.knowledge_builder` implements a manual, pre-run population workflow. It uses a
capable configured LLM in two stages: category-level topic planning followed by one focused Markdown
generation call per card. IDs and paths are derived by system code rather than accepted from the
model. The curator chooses the number of cards per category within configurable minimum and maximum
bounds, based on the number of distinct high-value topics rather than padding every category to a
fixed count. Metadata, duplicate topics, card length, title, and standard headings are validated
with bounded LLM repair attempts.

For OpenAI HTTP generation, the builder enables the Responses API `web_search` tool with required
tool choice and high search context by default. Category planning and every card-writing request are
therefore web-grounded. Search calls, actions, source lists, and URL citation annotations are kept in
the LLM audit, and validated citations are appended to each generated card. Normal ParLLiaMent runs do not
enable this tool. Command-based adapters must provide their own search implementation and audit.

The builder creates a complete sibling staging directory, regenerates `catalog.jsonl` and
`manifest.json`, and validates the staged result with the same `KnowledgeCatalog` used at runtime.
Only a valid result is installed, while the former knowledge directory is preserved as a timestamped
backup. It also retains a full request/response JSONL audit and a generation report. This utility is
available as `parlliament-populate-knowledge` and `scripts/populate_knowledge_base.py`; it is not imported
or invoked by the Overseer, Librarian, or normal run CLI. Extend mode is the default. Explicit
replacement mode retains repository-authored task, dataset, leakage, and evaluation cards while
rebuilding the retrievable research catalog. Task and dataset are not LLM-generated categories.
