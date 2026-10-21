from __future__ import annotations

from contextgcg import models
from contextgcg.generate import Generator
from contextgcg.optim.multigoal import MultiGoalConfig, MultiGoalSearch
from contextgcg.prompting import GCG_SLOT, doc, user


def goals(entry: dict, eot: str, single: bool) -> tuple[list, list[float]]:
    p = entry["passage"]
    g = [(user(doc(p, entry["base_question"], GCG_SLOT)), f"{entry['planted']}.")]
    if single:
        return g, [1.0]
    g += [(user(doc(p, t["question"], GCG_SLOT)), t["target"] + eot) for t in entry["train"]]
    k = len(entry["train"])
    return g, [max(float(k), 1.0)] + [1.0] * k


def run(model_name: str, entry: dict, single: bool = False, seed: int = 0, model=None, tok=None,
        steps: int = 250) -> dict:
    spec = models.MODELS[model_name]
    if model is None:
        model, tok = models.load(spec)
    cfg = MultiGoalConfig(steps=steps, search_batch_size=64, seed=seed)
    gl, w = goals(entry, spec.eot, single)
    curve = []
    suffix, loss = MultiGoalSearch(model, tok, spec.prefix_cache, cfg)(
        gl, w, on_step=lambda s, cur, best: curve.append((s, round(cur, 4), round(best, 4)))
        if s % 5 == 0 or s == cfg.steps - 1 else None)
    gen = Generator(model, tok, max_new_tokens=256, batch_size=1)
    test = gen([user(doc(entry["passage"], t["question"], suffix)) for t in entry["test"]])
    return {"item_id": entry["id"], "suffix": suffix, "loss": loss, "curve": curve,
            "test": [dict(question=t["question"], answer=a) for t, a in zip(entry["test"], test)]}
