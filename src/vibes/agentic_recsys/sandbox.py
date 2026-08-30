from __future__ import annotations

import json
import re
import shutil
import uuid
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


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


_HUNK = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


def apply_unified_diff(root: Path, expected_file: str, patch: str) -> None:
    """Apply a strict single-file unified diff without invoking a shell."""
    if expected_file not in MANAGED_FILES:
        raise GuardrailViolation(f"file is not agent-managed: {expected_file}")
    lines = patch.splitlines(keepends=True)
    if len(lines) < 2 or not lines[0].startswith("--- ") or not lines[1].startswith("+++ "):
        raise ValueError("patch must begin with unified diff file headers")

    def header_name(line: str) -> str:
        value = line[4:].strip().split("\t", 1)[0].replace("\\", "/")
        if value.startswith(("a/", "b/")):
            value = value[2:]
        return value

    old_name, new_name = header_name(lines[0]), header_name(lines[1])
    if old_name != expected_file or new_name != expected_file:
        raise GuardrailViolation(f"patch targets {old_name!r}->{new_name!r}, expected {expected_file!r}")
    path = guarded_path(root, expected_file)
    original = path.read_text(encoding="utf-8").splitlines(keepends=True)
    output: List[str] = []
    source_index = 0
    index = 2
    saw_hunk = False
    while index < len(lines):
        match = _HUNK.match(lines[index])
        if not match:
            if lines[index].strip() == "@@" and not any(
                remaining.strip() for remaining in lines[index + 1:]
            ):
                break
            if lines[index].strip():
                raise ValueError(f"unexpected patch line: {lines[index][:100]!r}")
            index += 1
            continue
        saw_hunk = True
        old_start = int(match.group(1)) - 1
        if old_start < source_index or old_start > len(original):
            raise ValueError("invalid or overlapping hunk location")
        output.extend(original[source_index:old_start])
        source_index = old_start
        index += 1
        while index < len(lines) and not lines[index].startswith("@@ "):
            line = lines[index]
            if line.startswith("\\ No newline"):
                index += 1
                continue
            if line.strip() == "@@" and not any(
                remaining.strip() for remaining in lines[index + 1:]
            ):
                index = len(lines)
                break
            if not line or line[0] not in " +-":
                raise ValueError(f"invalid hunk line: {line[:100]!r}")
            prefix, content = line[0], line[1:]
            if prefix in " -":
                if source_index >= len(original) or original[source_index].rstrip("\r\n") != content.rstrip("\r\n"):
                    raise ValueError("patch context does not match reference file")
                source_index += 1
            if prefix in " +":
                output.append(content)
            index += 1
    if not saw_hunk:
        raise ValueError("patch contains no hunks")
    output.extend(original[source_index:])
    path.write_text("".join(output), encoding="utf-8")


def apply_agent_patches(root: Path, patches: Dict[str, str], allowed_files: Iterable[str]) -> None:
    allowed = set(allowed_files)
    if not patches:
        raise ValueError("agent returned no patches")
    unexpected = set(patches) - allowed
    if unexpected:
        raise GuardrailViolation(f"agent attempted disallowed files: {sorted(unexpected)}")
    for filename, patch in patches.items():
        apply_unified_diff(root, filename, patch)


def config_diff(patches: Dict[str, str]) -> str:
    return patches.get("config.json", "")
