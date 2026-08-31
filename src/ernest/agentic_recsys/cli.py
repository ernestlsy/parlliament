from __future__ import annotations

import argparse
import json
import shlex
from pathlib import Path

from .config import SystemConfig
from .journal import Journal
from .llm import CommandLLMClient, OpenAICompatibleClient
from .overseer import Overseer


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ernest", description="Autonomous recommender MLE loop")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="start or resume a sequential run")
    run.add_argument("--workspace", required=True, help="directory that will contain runs/")
    run.add_argument("--data-dir", required=True, help="KuaiRand-Pure data directory")
    run.add_argument("--run-name", default="run_1")
    run.add_argument(
        "--seed-model",
        choices=["simple", "kuairand-baseline"],
        default="simple",
        help=(
            "parent-0 scaffold: simple additive user/item model (default), or the "
            "five-field KuaiRand Factorization Machine baseline"
        ),
    )
    providers = run.add_mutually_exclusive_group(required=True)
    providers.add_argument("--llm-command", help="local command accepting JSON stdin and emitting JSON stdout")
    providers.add_argument("--model", help="model for an OpenAI or compatible HTTP API")
    run.add_argument("--base-url", default="https://api.openai.com/v1")
    run.add_argument("--api-key", default=None)
    run.add_argument(
        "--api-mode", choices=["auto", "responses", "chat"], default="auto",
        help="API endpoint style; auto uses Responses for api.openai.com and Chat Completions elsewhere",
    )
    run.add_argument(
        "--no-json-mode", action="store_true",
        help="omit provider-side JSON mode fields for compatible providers that do not support them",
    )
    run.add_argument(
        "--llm-timeout", type=int, default=300,
        help="read timeout in seconds for each LLM HTTP request",
    )
    run.add_argument(
        "--llm-retries", type=int, default=2,
        help="HTTP retries for transient timeouts, network failures, rate limits, and server errors",
    )
    run.add_argument("--max-experiments", type=int, default=50)
    run.add_argument("--timeout", type=int, default=900)
    run.add_argument("--max-debug-attempts", type=int, default=3)
    run.add_argument("--max-backfills", type=int, default=2)
    run.add_argument("--candidate-pool-size", type=int, default=12)
    run.add_argument("--screening-timeout", type=int, default=900)
    run.add_argument("--screening-holdout-fraction", type=float, default=0.25)
    run.add_argument("--screening-seed", type=int, default=0)
    run.add_argument("--force-rescreen", action="store_true")
    run.add_argument(
        "--disable-literature", action="store_true",
        help="skip Librarian research requests and use only fixed knowledge cards",
    )
    run.add_argument("--literature-rounds", type=int, default=2)
    run.add_argument("--literature-max-documents", type=int, default=8)
    run.add_argument("--literature-character-budget", type=int, default=40_000)

    status = sub.add_parser("status", help="summarize a run journal")
    status.add_argument("run_dir")
    return parser


def main(argv=None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "status":
        journal = Journal(Path(args.run_dir) / "journal.jsonl")
        records = journal.records()
        scored = journal.scored()
        abandoned = [record for record in records if record["status"] == "abandoned"]
        research_path = Path(args.run_dir) / "research" / "screening_report.json"
        research_summary = {
            "ready": research_path.is_file(),
            "screening_report": str(research_path.resolve()),
            "feature_catalog": str(
                (Path(args.run_dir) / "research" / "feature_catalog.json").resolve()
            ),
            "status": None,
            "top_feature_groups": [],
        }
        if research_path.is_file():
            report = json.loads(research_path.read_text(encoding="utf-8"))
            research_summary["status"] = report.get("status")
            research_summary["top_feature_groups"] = [{
                "evidence_id": item.get("evidence_id"),
                "primary_lift": item.get("primary_lift"),
                "recommendation": item.get("recommendation"),
            } for item in report.get("candidates", [])[:5]]
        literature_manifests = sorted(
            Path(args.run_dir).glob("planning/generation_*/literature/retrieval_manifest.json"),
            key=lambda path: int(path.parent.parent.name.rsplit("_", 1)[-1]),
        )
        literature_summary = {
            "generations_with_retrieval": len(literature_manifests),
            "latest_manifest": None,
            "latest_selected_document_ids": [],
            "latest_fetched_characters": 0,
        }
        if literature_manifests:
            latest_literature = literature_manifests[-1]
            manifest = json.loads(latest_literature.read_text(encoding="utf-8"))
            literature_summary.update({
                "latest_manifest": str(latest_literature.resolve()),
                "latest_selected_document_ids": manifest.get("selected_document_ids", []),
                "latest_fetched_characters": manifest.get("total_fetched_characters", 0),
            })
        print(json.dumps({
            "records": len(records),
            "scored": len(scored),
            "abandoned": len(abandoned),
            "latest_primary": scored[-1]["metrics"]["primary"] if scored else None,
            "converged": journal.converged(),
            "seed_model": (
                json.loads(
                    (Path(args.run_dir) / "system_config.json").read_text(encoding="utf-8")
                ).get("seed_model", "simple")
                if (Path(args.run_dir) / "system_config.json").is_file() else None
            ),
            "research": research_summary,
            "literature": literature_summary,
            "recent_failures": [{
                "attempt_id": record["attempt_id"],
                "generation": record["generation"],
                "failure_stage": record.get("failure_stage"),
                "failure_reason": record.get("failure_reason"),
                "sandbox": record.get("sandbox"),
            } for record in abandoned[-10:]],
        }, indent=2))
        return 0

    config = SystemConfig(
        workspace=args.workspace,
        data_dir=args.data_dir,
        run_name=args.run_name,
        seed_model=args.seed_model,
        max_experiments=args.max_experiments,
        experiment_timeout_seconds=args.timeout,
        max_debug_attempts=args.max_debug_attempts,
        max_backfills_per_slot=args.max_backfills,
        candidate_pool_size=args.candidate_pool_size,
        screening_timeout_seconds=args.screening_timeout,
        screening_holdout_fraction=args.screening_holdout_fraction,
        screening_seed=args.screening_seed,
        force_rescreen=args.force_rescreen,
        literature_enabled=not args.disable_literature,
        literature_max_rounds=args.literature_rounds,
        literature_max_documents=args.literature_max_documents,
        literature_character_budget=args.literature_character_budget,
    )
    if args.llm_command:
        llm = CommandLLMClient(
            shlex.split(args.llm_command), timeout_seconds=args.llm_timeout
        )
    else:
        llm = OpenAICompatibleClient(
            args.model,
            base_url=args.base_url,
            api_key=args.api_key,
            timeout_seconds=args.llm_timeout,
            max_retries=args.llm_retries,
            api_mode=args.api_mode,
            json_mode=not args.no_json_mode,
        )
    print(json.dumps(Overseer(config, llm).run(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
