from __future__ import annotations

import json
import time
from pathlib import Path

from contextgcg import models
from contextgcg.data import Item
from contextgcg.generate import Generator
from contextgcg.optim.gcg import SearchConfig, SuffixSearch
from contextgcg.prompting import doc, user


def search(model_name: str, items: list[Item], out: Path, seed: int = 0, steps: int = 100,
           model=None, tok=None) -> list[dict]:
    spec = models.MODELS[model_name]
    if model is None:
        model, tok = models.load(spec)
    s = SuffixSearch(model, tok, spec.eot, spec.prefix_cache, SearchConfig(steps=steps, seed=seed))
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {json.loads(l)["item_id"] for l in out.open()} if out.exists() else set()
    for it in items:
        if it.id in done:
            continue
        t = time.time()
        r = s(it)
        with out.open("a") as f:
            f.write(json.dumps({"item_id": it.id, "suffix": r.suffix, "loss": r.loss,
                                "losses": r.losses, "seconds": round(time.time() - t, 1)}) + "\n")
    return [json.loads(l) for l in out.open()]


def answers(model, tok, items: dict[str, Item], rows: list[dict], batch_size: int = 8,
            max_new_tokens: int = 256) -> list[str]:
    gen = Generator(model, tok, max_new_tokens, batch_size)
    return gen([user(doc(items[r["item_id"]].passage, items[r["item_id"]].query, r["suffix"]))
                for r in rows])
