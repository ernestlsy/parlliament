# Ernest research knowledge base

Markdown cards are the source of truth. `catalog.jsonl` is the only discovery interface used by
the Librarian, and document IDs—not paths—are the public retrieval identifiers. Small cards marked
`always_include` provide fixed task, dataset, and safety context; all other cards are retrieved on
demand. Repository-authored task and dataset cards are marked `system_owned`, so replacement builds
preserve them without forcing every detailed dataset card into every prompt.

Every card describes one method family using the same practical headings: mechanism, applicability,
requirements, implementation, starting configuration, expected metric effects, diagnostics, risks,
cheap preliminary check, clean experiment, related cards, and sources.

The `task` and `dataset` categories are maintained from Ernest's specification, immutable evaluator,
and direct measurements of the supplied KuaiRand-Pure CSVs. The one-time knowledge builder generates
only the remaining literature/research categories.
