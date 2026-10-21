from __future__ import annotations

import nanogcg
from nanogcg import GCGConfig

from contextgcg import models
from contextgcg.data import Item
from contextgcg.generate import Generator
from contextgcg.optim.multigoal import MultiGoalConfig, MultiGoalSearch
from contextgcg.prompting import GCG_SLOT, doc, reask as reask_msgs, user


def reask(model, tok, turn1: list[tuple[str, str, str]], batch_size: int = 16,
          max_new_tokens: int = 256, stop_at_eot: str | None = None) -> list[tuple[str, str]]:
    gen = Generator(model, tok, max_new_tokens, batch_size, stop_at_eot)
    h2 = [reask_msgs(u, a.strip(), q) for u, a, q in turn1]
    t2 = gen(h2)
    t3 = gen([h + [{"role": "assistant", "content": a}, {"role": "user", "content": h[-1]["content"]}]
              for h, a in zip(h2, t2)])
    return list(zip(t2, t3))


def repair(model_name: str, item: Item, target: str, condition: str, steps: int = 150, seed: int = 1,
           search_batch_size: int = 128, model=None, tok=None) -> dict:
    spec = models.MODELS[model_name]
    if model is None:
        model, tok = models.load(spec)
    tgt = target.format(ck=item.planted)
    t1 = user(doc(item.passage, item.query, GCG_SLOT))
    if condition == "joint":
        t2 = t1 + [{"role": "assistant", "content": tgt}, {"role": "user", "content": item.query}]
        cfg = MultiGoalConfig(steps=steps, search_batch_size=search_batch_size, seed=seed)
        suffix, loss = MultiGoalSearch(model, tok, spec.prefix_cache, cfg)([(t1, tgt), (t2, tgt)])
    elif condition == "control":
        r = nanogcg.run(model, tok, t1, tgt, GCGConfig(
            num_steps=steps, optim_str_init=" ".join(["x"] * 20), search_width=512, topk=256,
            early_stop=False, add_space_before_target=False, seed=seed, verbosity="WARNING",
            use_prefix_cache=spec.prefix_cache))
        suffix, loss = r.best_string, r.best_loss
    else:
        raise ValueError(condition)
    gen = Generator(model, tok, max_new_tokens=256, batch_size=1)
    u1 = doc(item.passage, item.query, suffix)
    a1 = gen([user(u1)])[0]
    h = reask_msgs(u1, a1, item.query)
    a2 = gen([h])[0]
    a3 = gen([h + [{"role": "assistant", "content": a2}, {"role": "user", "content": item.query}]])[0]
    return {"item_id": item.id, "condition": condition, "suffix": suffix, "loss": float(loss),
            "turn1": a1, "turn2": a2, "turn3": a3}
