from __future__ import annotations

import json
from pathlib import Path

from contextgcg import models
from contextgcg.data import Item
from contextgcg.generate import Generator
from contextgcg.optim.universal import UniversalConfig, UniversalSearch
from contextgcg.prompting import doc, user


def search(model_name: str, train: list[Item], cfg: UniversalConfig, out: Path,
           model=None, tok=None) -> dict:
    spec = models.MODELS[model_name]
    if model is None:
        model, tok = models.load(spec)
    out.mkdir(parents=True, exist_ok=True)

    def checkpoint(step, suffix, best):
        (out / "checkpoint.json").write_text(json.dumps({"step": step, "suffix": suffix,
                                                         "select_loss": best}))
    r = UniversalSearch(model, tok, spec.prefix_cache, cfg)(train, on_eval=checkpoint)
    res = {"suffix": r.suffix, "select_loss": r.select_loss, "curve": r.curve,
           "schedule": r.schedule, "train_ids": [it.id for it in train]}
    (out / "suffix.json").write_text(json.dumps(res, indent=1))
    return res


def answers(model, tok, items: list[Item], suffix: str, batch_size: int = 16,
            max_new_tokens: int = 256) -> list[str]:
    gen = Generator(model, tok, max_new_tokens, batch_size)
    return gen([user(doc(it.passage, it.query, suffix)) for it in items])
