from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Item:
    id: str
    query: str
    passage: str
    planted: str
    true: str
    meta: dict = field(default_factory=dict, hash=False, compare=False)


def load_items(path: str | Path) -> list[Item]:
    with open(path) as f:
        rows = [json.loads(line) for line in f if line.strip()]
    return [Item(r["id"], r["query"], r["passage"], r["planted"], r["true"], r.get("meta", {}))
            for r in rows]
