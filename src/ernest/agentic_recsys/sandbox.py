from __future__ import annotations

import ast
import json
import shutil
import uuid
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple


MANAGED_FILES = {"data.py", "model.py", "train.py", "config.json"}


class GuardrailViolation(ValueError):
    pass


def guarded_path(root: Path, relative: str) -> Path:
    if Path(relative).is_absolute():
        raise GuardrailViolation("absolute paths are forbidden")
    target = (root / relative).resolve()
    root = root.resolve()
    try:
        target.relative_to(root)
    except ValueError as exc:
        raise GuardrailViolation(f"path escapes sandbox: {relative}") from exc
    return target


def create_attempt_sandbox(run_dir: Path, parent: Path) -> Tuple[str, Path]:
    attempt_id = uuid.uuid4().hex[:12]
    staging = run_dir / "attempts" / f"attempt_{attempt_id}"
    staging.parent.mkdir(parents=True, exist_ok=True)
    staging.mkdir()
    for name in MANAGED_FILES:
        source = guarded_path(parent, name)
        if not source.is_file():
            raise FileNotFoundError(f"parent is missing {name}")
        shutil.copy2(source, guarded_path(staging, name))
    return attempt_id, staging


def finalize_sandbox(
    staging: Path, run_dir: Path, *, experiment_id: "Optional[int]", attempt_id: str
) -> Path:
    if experiment_id is None:
        destination = run_dir / "abandoned" / f"attempt_{attempt_id}"
    else:
        destination = run_dir / f"experiment_{experiment_id}"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"sandbox already exists: {destination}")
    staging.rename(destination)
    return destination


def apply_agent_replacements(
    root: Path, replacements: Dict[str, str], allowed_files: Iterable[str]
) -> None:
    """Validate and transactionally replace every file owned by one agent."""
    allowed = set(allowed_files)
    if not allowed or not allowed.issubset(MANAGED_FILES):
        raise GuardrailViolation(f"invalid managed-file assignment: {sorted(allowed)}")
    received = set(replacements)
    unexpected = received - allowed
    if unexpected:
        raise GuardrailViolation(f"agent attempted disallowed files: {sorted(unexpected)}")
    missing = allowed - received
    if missing:
        raise ValueError(f"agent response is missing complete files: {sorted(missing)}")

    paths = {filename: guarded_path(root, filename) for filename in allowed}
    originals = {}
    for filename, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"managed file does not exist: {filename}")
        originals[filename] = path.read_text(encoding="utf-8")

    for filename in sorted(allowed):
        content = replacements[filename]
        if not isinstance(content, str):
            raise ValueError(f"complete content for {filename} must be a string")
        if not content.strip():
            raise ValueError(f"complete content for {filename} cannot be empty")
        if "\x00" in content:
            raise ValueError(f"complete content for {filename} contains a null byte")
        if filename.endswith(".py"):
            try:
                ast.parse(content, filename=filename)
            except (SyntaxError, UnicodeError) as exc:
                location = f"line {exc.lineno}" if isinstance(exc, SyntaxError) else "unknown line"
                raise ValueError(f"invalid Python in {filename} at {location}: {exc}") from exc
        elif filename == "config.json":
            try:
                config = json.loads(content)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"invalid JSON in config.json at line {exc.lineno}, column {exc.colno}: {exc.msg}"
                ) from exc
            if not isinstance(config, dict):
                raise ValueError("config.json must contain a JSON object")

    if all(replacements[name] == originals[name] for name in allowed):
        raise ValueError("agent returned complete files with no changes")

    written = []
    try:
        for filename in sorted(allowed):
            written.append(filename)
            paths[filename].write_text(replacements[filename], encoding="utf-8")
    except Exception:
        for filename in written:
            paths[filename].write_text(originals[filename], encoding="utf-8")
        raise


def config_diff(diffs: Dict[str, str]) -> str:
    return diffs.get("config.json", "")
