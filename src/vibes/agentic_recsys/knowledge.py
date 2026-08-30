from __future__ import annotations

from pathlib import Path
from typing import Dict, List


class KnowledgeBase:
    """Read-only local literature/research context supplied to the Evolution Judge."""

    SUPPORTED_SUFFIXES = {".md", ".txt", ".json"}

    def __init__(self, root: Path, max_characters: int = 80_000):
        self.root = root
        self.max_characters = max_characters

    def documents(self) -> List[Dict[str, str]]:
        if not self.root.is_dir():
            return []
        documents = []
        remaining = self.max_characters
        for path in sorted(self.root.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in self.SUPPORTED_SUFFIXES:
                continue
            text = path.read_text(encoding="utf-8")
            content = text[:remaining]
            documents.append({
                "source": str(path.relative_to(self.root)).replace("\\", "/"),
                "content": content,
                "truncated": str(len(content) < len(text)).lower(),
            })
            remaining -= len(content)
            if remaining <= 0:
                break
        return documents

