from __future__ import annotations

import copy
from dataclasses import dataclass

import torch
from nanogcg import GCGConfig
from nanogcg.gcg import GCG, filter_ids, sample_ids_from_grad
from nanogcg.utils import find_executable_batch_size

from contextgcg.prompting import GCG_SLOT


@dataclass(frozen=True)
class MultiGoalConfig:
    steps: int
    topk: int = 256
    search_width: int = 512
    search_batch_size: int = 64
    suffix_len: int = 20
    seed: int = 0


class MultiGoalSearch:
    def __init__(self, model, tok, prefix_cache: bool, cfg: MultiGoalConfig) -> None:
        self.model, self.tok, self.cfg, self.prefix_cache = model, tok, cfg, prefix_cache
        init = " ".join(["x"] * cfg.suffix_len)
        assert len(tok(init, add_special_tokens=False)["input_ids"]) == cfg.suffix_len
        self.gcg = GCG(model, tok, GCGConfig(num_steps=cfg.steps, search_width=cfg.search_width,
                                             topk=cfg.topk, n_replace=1, optim_str_init=init,
                                             filter_ids=True, early_stop=False, verbosity="ERROR"))
        self.init = init

    def goal_ids(self, messages: list[dict], target: str) -> tuple:
        tok = self.tok
        messages = copy.deepcopy(messages)
        if not any(GCG_SLOT in m["content"] for m in messages):
            messages[-1]["content"] += GCG_SLOT
        text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        if tok.bos_token and text.startswith(tok.bos_token):
            text = text.replace(tok.bos_token, "")
        before, after = text.split(GCG_SLOT)
        dev = self.model.device
        ids = lambda s, **kw: tok([s], return_tensors="pt", **kw)["input_ids"].to(dev, torch.int64)
        return ids(before, padding=False), ids(after, add_special_tokens=False), \
            ids(target, add_special_tokens=False)

    def goal(self, messages: list[dict], target: str) -> dict:
        before, after, target_ids = self.goal_ids(messages, target)
        emb = self.gcg.embedding_layer
        g = dict(before_embeds=emb(before), after_embeds=emb(after), target_embeds=emb(target_ids),
                 target_ids=target_ids, prefix_cache=None)
        if self.prefix_cache:
            with torch.no_grad():
                g["prefix_cache"] = self.model(inputs_embeds=g["before_embeds"],
                                               use_cache=True).past_key_values
        return g

    def _activate(self, g: dict) -> None:
        for k in ("before_embeds", "after_embeds", "target_embeds", "target_ids", "prefix_cache"):
            setattr(self.gcg, k, g[k])

    def __call__(self, goals: list[tuple[list[dict], str]], weights: list[float] | None = None,
                 on_step=None) -> tuple[str, float]:
        c, tok, gcg = self.cfg, self.tok, self.gcg
        torch.manual_seed(c.seed)
        torch.cuda.manual_seed_all(c.seed)
        prepared = [self.goal(m, t) for m, t in goals]
        wts = [1.0] * len(prepared) if weights is None else list(weights)
        assert len(wts) == len(prepared) and all(w > 0 for w in wts)

        optim_ids = tok(self.init, add_special_tokens=False, return_tensors="pt")["input_ids"]
        optim_ids = optim_ids.to(self.model.device)
        best_loss, best_ids = float("inf"), optim_ids
        for step in range(c.steps):
            grad = None
            for w, g in zip(wts, prepared):
                self._activate(g)
                gl = w * gcg.compute_token_gradient(optim_ids)
                grad = gl if grad is None else grad + gl
            with torch.no_grad():
                cand = sample_ids_from_grad(optim_ids.squeeze(0), grad.squeeze(0), c.search_width,
                                            c.topk, 1, not_allowed_ids=gcg.not_allowed_ids)
                cand = filter_ids(cand, tok)
                width, total = cand.shape[0], None
                for w, g in zip(wts, prepared):
                    self._activate(g)
                    parts = [] if g["prefix_cache"] is not None else [g["before_embeds"].repeat(width, 1, 1)]
                    embeds = torch.cat(parts + [gcg.embedding_layer(cand),
                                                g["after_embeds"].repeat(width, 1, 1),
                                                g["target_embeds"].repeat(width, 1, 1)], dim=1)
                    loss = find_executable_batch_size(gcg.compute_candidates_loss,
                                                      c.search_batch_size)(embeds)
                    total = w * loss if total is None else total + w * loss
                cur, win = total.min().item(), total.argmin()
                optim_ids = cand[win].unsqueeze(0)
                if cur < best_loss:
                    best_loss, best_ids = cur, optim_ids
            if on_step is not None:
                on_step(step, cur, best_loss)
        return tok.batch_decode(best_ids)[0], best_loss
