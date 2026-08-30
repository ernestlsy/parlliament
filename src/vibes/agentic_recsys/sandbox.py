from __future__ import annotations

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


# Search/replace blocks replaced unified diffs as the agent-facing edit format. Every
# abandonment in run_2 was "patch context does not match reference file" with correct
# code inside the rejected patch: models are unreliable at hunk headers and line-count
# arithmetic, and reliable at quoting a span of code they were just shown. The journal
# still stores a unified diff, computed against the parent after the fact.
_BLOCK_START = re.compile(r"^<{5,}\s*(?:SEARCH)?\s*$")
_BLOCK_DIVIDER = re.compile(r"^={5,}\s*$")
_BLOCK_END = re.compile(r"^>{5,}\s*(?:REPLACE)?\s*$")

BLOCK_FORMAT_HELP = (
    "Each edit is a block of the form:\n"
    "<<<<<<< SEARCH\n"
    "<exact text copied from the current file>\n"
    "=======\n"
    "<replacement text>\n"
    ">>>>>>> REPLACE\n"
    "SEARCH must reproduce the current file exactly and must appear exactly once in it; "
    "include surrounding lines to make it unique. Emit several blocks to make several "
    "edits."
)


def _preview(text: str) -> str:
    lines = text.splitlines() or [""]
    extra = "" if len(lines) == 1 else f" (+{len(lines) - 1} more lines)"
    return repr(lines[0].strip()[:120]) + extra


def parse_search_replace(patch: str) -> List[Tuple[str, str]]:
    """Parse SEARCH/REPLACE blocks; text outside a block is commentary and is ignored."""
    lines = patch.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    blocks: List[Tuple[str, str]] = []
    index = 0
    while index < len(lines):
        if not _BLOCK_START.match(lines[index]):
            index += 1
            continue
        start_line = index + 1
        index += 1
        search: List[str] = []
        while index < len(lines) and not _BLOCK_DIVIDER.match(lines[index]):
            if _BLOCK_START.match(lines[index]) or _BLOCK_END.match(lines[index]):
                raise ValueError(
                    f"SEARCH block opened at line {start_line} is missing its '=======' "
                    f"divider. {BLOCK_FORMAT_HELP}"
                )
            search.append(lines[index])
            index += 1
        if index >= len(lines):
            raise ValueError(
                f"SEARCH block opened at line {start_line} is missing its '=======' "
                f"divider. {BLOCK_FORMAT_HELP}"
            )
        index += 1
        replace: List[str] = []
        while index < len(lines) and not _BLOCK_END.match(lines[index]):
            if _BLOCK_START.match(lines[index]) or _BLOCK_DIVIDER.match(lines[index]):
                raise ValueError(
                    f"block opened at line {start_line} is missing its '>>>>>>> REPLACE' "
                    f"terminator. {BLOCK_FORMAT_HELP}"
                )
            replace.append(lines[index])
            index += 1
        if index >= len(lines):
            raise ValueError(
                f"block opened at line {start_line} is missing its '>>>>>>> REPLACE' "
                f"terminator. {BLOCK_FORMAT_HELP}"
            )
        index += 1
        blocks.append(("\n".join(search), "\n".join(replace)))
    if not blocks:
        raise ValueError(f"patch contains no SEARCH/REPLACE blocks. {BLOCK_FORMAT_HELP}")
    return blocks


def _apply_block(content: str, search: str, replace: str, expected_file: str) -> str:
    if not search.strip():
        raise ValueError(
            f"empty SEARCH text for {expected_file}; SEARCH must quote the exact current "
            "text being replaced. To rewrite the file wholesale, SEARCH its full contents."
        )
    if search == replace:
        raise ValueError(
            f"SEARCH and REPLACE are identical for {expected_file}; the block is a no-op"
        )
    occurrences = content.count(search)
    if occurrences == 1:
        return content.replace(search, replace, 1)
    if occurrences > 1:
        raise ValueError(
            f"SEARCH text matched {occurrences} times in {expected_file}; it must match "
            f"exactly once. Extend the block with surrounding lines to make it unique. "
            f"First line was {_preview(search)}"
        )
    # Second tier: whole-line match ignoring trailing whitespace, which models drop.
    # Still literal, and still required to be unique, so it cannot silently pick a hit.
    file_lines = content.split("\n")
    search_lines = [line.rstrip() for line in search.split("\n")]
    stripped = [line.rstrip() for line in file_lines]
    span = len(search_lines)
    matches = [
        start for start in range(len(stripped) - span + 1)
        if stripped[start:start + span] == search_lines
    ]
    if len(matches) == 1:
        start = matches[0]
        return "\n".join(
            file_lines[:start] + replace.split("\n") + file_lines[start + span:]
        )
    if len(matches) > 1:
        raise ValueError(
            f"SEARCH text matched {len(matches)} places in {expected_file} once trailing "
            f"whitespace is ignored; it must match exactly once. Extend the block with "
            f"surrounding lines. First line was {_preview(search)}"
        )
    raise ValueError(
        f"SEARCH text was not found in {expected_file}; it must reproduce the current file "
        f"exactly, character for character, from the copy supplied in current_files. "
        f"First line was {_preview(search)}"
    )


def apply_search_replace(root: Path, expected_file: str, patch: str) -> None:
    """Apply literal SEARCH/REPLACE blocks to one agent-managed file."""
    if expected_file not in MANAGED_FILES:
        raise GuardrailViolation(f"file is not agent-managed: {expected_file}")
    path = guarded_path(root, expected_file)
    content = path.read_text(encoding="utf-8")
    for search, replace in parse_search_replace(patch):
        content = _apply_block(content, search, replace, expected_file)
    path.write_text(content, encoding="utf-8", newline="\n")


def apply_agent_patches(root: Path, patches: Dict[str, str], allowed_files: Iterable[str]) -> None:
    allowed = set(allowed_files)
    if not patches:
        raise ValueError("agent returned no patches")
    unexpected = set(patches) - allowed
    if unexpected:
        raise GuardrailViolation(f"agent attempted disallowed files: {sorted(unexpected)}")
    for filename, patch in patches.items():
        apply_search_replace(root, filename, patch)


def config_diff(patches: Dict[str, str]) -> str:
    return patches.get("config.json", "")
