from __future__ import annotations

import argparse
import json
import sys
from functools import cached_property, lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from contextgcg import models                                   # noqa: E402
from contextgcg.data import Item, load_items                    # noqa: E402
from contextgcg.prompting import render                         # noqa: E402
from contextgcg.io import read_jsonl, write_jsonl            # noqa: E402

DATA = ROOT / "data"
MODEL_NAMES = tuple(models.MODELS)


def parser(description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, choices=MODEL_NAMES)
    p.add_argument("--limit", type=int, default=0, help="first N items only")
    p.add_argument("--out", type=Path, default=ROOT / "out", help="output root (default ./out)")
    p.add_argument("--dry-run", action="store_true",
                   help="build the prompts only; no model, no GPU")
    p.add_argument("--judge", action="store_true",
                   help="also score answers with the Qwen3-32B cross-check (needs requirements-judge.txt)")
    return p


def items(name: str, model: str) -> list[Item]:
    return load_items(DATA / "items" / name / f"{model}.jsonl")


def released(section: str, model: str, condition: str) -> list[dict]:
    path = DATA / "runs" / section / model / f"{condition}.jsonl"
    if not path.exists():
        raise SystemExit(f"no released condition {section}/{model}/{condition} (see `python reproduce.py conditions`)")
    return read_jsonl(path)


def clip(xs: list, limit: int) -> list:
    return xs[:limit] if limit else xs


class Session:
    def __init__(self, args) -> None:
        self.args, self.spec = args, models.MODELS[args.model]

    @cached_property
    def tok(self):
        return models.load_tokenizer(self.spec)

    @cached_property
    def model_tok(self):
        return models.load(self.spec)

    @property
    def model(self):
        return self.model_tok[0]

    def generate(self, conversations: list[list[dict]], batch_size: int = 8, max_new_tokens: int = 256,
                 stop_at_eot: bool = False, hooks=None) -> list[str]:
        if self.args.dry_run:
            show_prompt(self.tok, conversations)
            return ["<dry-run>"] * len(conversations)
        from contextgcg.generate import Generator
        model, tok = self.model_tok
        gen = Generator(model, tok, max_new_tokens, batch_size, self.spec.eot if stop_at_eot else None)
        return gen(conversations, hooks=hooks)


def show_prompt(tok, conversations: list[list[dict]]) -> None:
    if not conversations:
        print("[dry-run] 0 prompts built: this condition has nothing to generate.")
        return
    print(f"[dry-run] {len(conversations)} prompts built. First one, exactly as the model sees it:")
    print("-" * 80)
    print(render(tok, conversations[0]))
    print("-" * 80)


def out_dir(args, rq: str) -> Path:
    d = args.out / rq / args.model
    d.mkdir(parents=True, exist_ok=True)
    return d


def save(path: Path, rows: list[dict]) -> None:
    write_jsonl(path, rows)
    print(f"wrote {len(rows)} answers -> {path}")


@lru_cache(maxsize=None)
def _qwen(rubric: str):
    from contextgcg.judge import JUDGE_REPO, QwenJudge
    net, tok = models.load(models.ModelSpec("qwen", JUDGE_REPO, "", False, None))
    return QwenJudge(net, tok, rubric)


def judge(args, rows: list[dict], items_by_id: dict[str, Item], rubric: str = "flip",
          answer_key: str = "answer") -> None:
    if not args.judge or args.dry_run:
        return
    J = _qwen(rubric)
    for r in rows:
        it = items_by_id[r["item_id"]]
        r["qwen"] = J(passage=r.get("passage", it.passage), query=r.get("question", it.query),
                      planted=r.get("target", it.planted), true=it.true, answer=r[answer_key],
                      claim=it.meta.get("claim", ""))
    n = sum(r["qwen"] for r in rows)
    print(f"Qwen3-32B cross-check: {n}/{len(rows)} = {n / len(rows):.3f} follow the passage")



def dump(obj) -> str:
    return json.dumps(obj, ensure_ascii=False)
