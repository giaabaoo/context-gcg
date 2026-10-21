from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from contextgcg.io import read_jsonl


class ReadingError(Exception):
    pass


def fill(rubric: str, claim: str, query: str, target: str, answer: str) -> str:
    return (rubric.replace("{claim}", claim).replace("{query}", query)
            .replace("{ck}", target).replace("{text}", answer))


def from_condition(release: Path, condition: str) -> list[dict]:
    section, model, name = condition.split("/")
    if section != "entailed":
        raise ReadingError(f"{condition}: this rubric is for the entailed conditions, not {section!r}")
    claims = {it["id"]: it.get("meta", {}).get("claim", "")
              for it in read_jsonl(release / "items" / "entailed" / f"{model}.jsonl")}
    path = release / "runs" / section / model / f"{name}.jsonl"
    if not path.exists():
        raise ReadingError(f"no condition {condition} (see `python reproduce.py conditions`)")
    out = []
    for i, r in enumerate(read_jsonl(path)):
        claim = claims.get(r["item_id"], "")
        if not claim:
            raise ReadingError(f"{condition} line {i}: no claim for item {r['item_id']}")
        out.append({"id": f"{condition}#{i}", "model": model, "condition": name,
                    "item_id": r["item_id"], "question_id": r.get("question_id"),
                    "claim": claim, "query": r.get("question"), "target": r.get("target"),
                    "answer": r["answer"], "qwen": (r.get("labels") or {}).get("qwen"),
                    "reading": None})
    return out


@lru_cache(maxsize=None)
def _rows(path: str) -> tuple:
    return tuple(read_jsonl(Path(path)))


def from_sample(release: Path) -> list[dict]:
    sample = release / "runs" / "entailed" / "kappa_sample.json"
    if not sample.exists():
        raise ReadingError(f"{sample} not found")
    d = json.loads(sample.read_text())
    claims: dict[str, str] = {}
    for m in ("llama", "mistral", "gemma"):
        for it in read_jsonl(release / "items" / "entailed" / f"{m}.jsonl"):
            claims[f"{m}/{it['id']}"] = it.get("meta", {}).get("claim", "")
    out = []
    for i, r in enumerate(d["rows"]):
        item_id = _rows(str(release.parent / r["file"]))[r["record_index"]]["item_id"]
        out.append({"id": f"sample#{i}", "model": r["model"], "condition": Path(r["file"]).stem,
                    "item_id": item_id, "question_id": None,
                    "claim": claims.get(f"{r['model']}/{item_id}", ""),
                    "query": r["question"], "target": r["target"], "answer": r["answer"],
                    "qwen": r["qwen"], "reading": r["reading"]})
    return out


def prompts(rubric: str, rows: list[dict]) -> list[dict]:
    return [{"id": r["id"], "prompt": fill(rubric, r["claim"], r["query"], r["target"],
                                           r["answer"])} for r in rows]


def verdicts(path: Path, n: int) -> list[bool]:
    raw = [l.strip() for l in path.read_text().splitlines() if l.strip()]
    if len(raw) != n:
        raise ReadingError(f"{path}: {len(raw)} verdicts for {n} prompts. One line per prompt, "
                           f"in the order written.")
    out = []
    for i, l in enumerate(raw):
        letter = l.split()[-1].strip().upper()[:1]
        if letter not in ("A", "B"):
            raise ReadingError(f"{path} line {i + 1}: expected A or B, got {l!r}")
        out.append(letter == "A")
    return out
