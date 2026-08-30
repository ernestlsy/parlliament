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
    run.add_argument("--max-experiments", type=int, default=50)
    run.add_argument("--timeout", type=int, default=900)
    run.add_argument("--max-debug-attempts", type=int, default=3)
    run.add_argument("--max-backfills", type=int, default=2)

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
        print(json.dumps({
            "records": len(records),
            "scored": len(scored),
            "abandoned": len(abandoned),
            "latest_primary": scored[-1]["metrics"]["primary"] if scored else None,
            "converged": journal.converged(),
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
        max_experiments=args.max_experiments,
        experiment_timeout_seconds=args.timeout,
        max_debug_attempts=args.max_debug_attempts,
        max_backfills_per_slot=args.max_backfills,
    )
    if args.llm_command:
        llm = CommandLLMClient(shlex.split(args.llm_command))
    else:
        llm = OpenAICompatibleClient(
            args.model,
            base_url=args.base_url,
            api_key=args.api_key,
            api_mode=args.api_mode,
            json_mode=not args.no_json_mode,
        )
    print(json.dumps(Overseer(config, llm).run(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
