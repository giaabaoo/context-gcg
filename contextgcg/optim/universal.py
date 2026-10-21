from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from nanogcg.gcg import filter_ids, sample_ids_from_grad
from nanogcg.utils import get_nonascii_toks

from contextgcg.data import Item
from contextgcg.prompting import GCG_SLOT, doc


@dataclass(frozen=True)
class UniversalConfig:
    target: str
    progressive: bool
    steps: int = 500
    topk: int = 256
    search_width: int = 256
    search_batch_size: int = 128
    eval_every: int = 25
    select_on: int = 8
    suffix_len: int = 20
    seed: int = 0


@dataclass
class UniversalResult:
    suffix: str
    select_loss: float
    curve: list = field(default_factory=list)
    schedule: list = field(default_factory=list)


@dataclass
class _Goal:
    after_embeds: torch.Tensor
    target_ids: torch.Tensor
    target_embeds: torch.Tensor
    prefix_kv: list
    before_embeds: torch.Tensor | None = None


class UniversalSearch:
    def __init__(self, model, tok, prefix_cache: bool, cfg: UniversalConfig) -> None:
        self.model, self.tok, self.cfg, self.prefix_cache = model, tok, cfg, prefix_cache
        self.dev = model.device
        self.emb = model.get_input_embeddings()
        self.not_allowed = get_nonascii_toks(tok, device=self.dev)

    def goal_strings(self, item: Item) -> tuple[str, str, str]:
        tok = self.tok
        text = tok.apply_chat_template([{"role": "user", "content": doc(item.passage, item.query, GCG_SLOT)}],
                                       tokenize=False, add_generation_prompt=True)
        if tok.bos_token and text.startswith(tok.bos_token):
            text = text.replace(tok.bos_token, "", 1)
        before, after = text.split(GCG_SLOT)
        return before, after, self.cfg.target.format(ck=item.planted)

    @torch.no_grad()
    def _goal(self, item: Item) -> _Goal:
        before, after, target = self.goal_strings(item)
        ids = lambda s, **kw: self.tok([s], return_tensors="pt", **kw)["input_ids"].to(self.dev)
        before_ids, after_ids = ids(before), ids(after, add_special_tokens=False)
        target_ids = ids(target, add_special_tokens=False)
        before_embeds = self.emb(before_ids)
        if not self.prefix_cache:
            return _Goal(self.emb(after_ids), target_ids, self.emb(target_ids), [], before_embeds)
        cache = self.model(inputs_embeds=before_embeds, use_cache=True).past_key_values
        kv = (list(zip(cache.key_cache, cache.value_cache)) if hasattr(cache, "key_cache")
              else [(k, v) for k, v in cache])
        return _Goal(self.emb(after_ids), target_ids, self.emb(target_ids), kv)

    def _forward(self, g: _Goal, parts: list, b: int):
        if not self.prefix_cache:
            embeds = torch.cat([g.before_embeds.expand(b, -1, -1)] + parts, dim=1)
            return self.model(inputs_embeds=embeds).logits, embeds.shape[1]
        embeds = torch.cat(parts, dim=1)
        kv = [[k.expand(b, -1, -1, -1), v.expand(b, -1, -1, -1)] for k, v in g.prefix_kv]
        return self.model(inputs_embeds=embeds, past_key_values=kv, use_cache=True).logits, embeds.shape[1]

    def _grad(self, optim_ids, g: _Goal) -> torch.Tensor:
        onehot = F.one_hot(optim_ids, self.emb.num_embeddings).to(self.model.dtype)
        onehot.requires_grad_()
        logits, n = self._forward(g, [onehot @ self.emb.weight, g.after_embeds, g.target_embeds], 1)
        shift = n - g.target_ids.shape[1]
        loss = F.cross_entropy(logits[:, shift - 1:-1].reshape(-1, logits.size(-1)), g.target_ids.reshape(-1))
        return torch.autograd.grad(loss, onehot)[0]

    @torch.no_grad()
    def _cand_loss(self, cand, g: _Goal) -> torch.Tensor:
        out = []
        for chunk in cand.split(self.cfg.search_batch_size):
            b = chunk.shape[0]
            logits, n = self._forward(g, [self.emb(chunk), g.after_embeds.repeat(b, 1, 1),
                                          g.target_embeds.repeat(b, 1, 1)], b)
            shift = n - g.target_ids.shape[1]
            sl = logits[:, shift - 1:-1].contiguous()
            ce = F.cross_entropy(sl.view(-1, sl.size(-1)), g.target_ids.repeat(b, 1).view(-1),
                                 reduction="none").view(b, -1).mean(1)
            out.append(ce)
        return torch.cat(out)

    def _mean_loss(self, cand, goals: list[_Goal]) -> torch.Tensor:
        total = torch.zeros(cand.shape[0], device=self.dev)
        for g in goals:
            total += self._cand_loss(cand, g)
        return total / len(goals)

    @torch.no_grad()
    def _solved(self, optim_ids, goals: list[_Goal]) -> bool:
        for g in goals:
            logits, n = self._forward(g, [self.emb(optim_ids), g.after_embeds, g.target_embeds], 1)
            shift = n - g.target_ids.shape[1]
            if int(logits[0, shift - 1].argmax()) != int(g.target_ids[0, 0]):
                return False
        return True

    def __call__(self, train: list[Item], log=print, on_eval=None) -> UniversalResult:
        c = self.cfg
        torch.manual_seed(c.seed)
        goals = [self._goal(it) for it in train]
        selection = goals[:min(c.select_on, len(goals))]
        init = " ".join(["x"] * c.suffix_len)
        optim_ids = self.tok(init, add_special_tokens=False, return_tensors="pt")["input_ids"].to(self.dev)
        best_ids, best, res = optim_ids.clone(), float("inf"), UniversalResult("", 0.0)
        n_active = 1 if c.progressive else len(goals)

        for step in range(c.steps):
            active = goals[:n_active]
            grad = sum(self._grad(optim_ids, g) for g in active)
            with torch.no_grad():
                cand = sample_ids_from_grad(optim_ids.squeeze(0), grad.squeeze(0), c.search_width,
                                            c.topk, n_replace=1, not_allowed_ids=self.not_allowed)
                cand = filter_ids(cand, self.tok)
                loss = self._mean_loss(cand, active)
                optim_ids = cand[loss.argmin()].unsqueeze(0)
                if c.progressive and n_active < len(goals) and self._solved(optim_ids, active):
                    n_active += 1
                    res.schedule.append((step, n_active))
            if step % c.eval_every == 0 or step == c.steps - 1:
                with torch.no_grad():
                    h = self._mean_loss(optim_ids, selection).item()
                res.curve.append(round(h, 4))
                if h < best:
                    best, best_ids = h, optim_ids.clone()
                log(f"step {step} | active {n_active}/{len(goals)} | select {h:.4f} | best {best:.4f}")
                if on_eval is not None:
                    on_eval(step, self.tok.decode(best_ids.squeeze(0)), best)
        res.suffix, res.select_loss = self.tok.decode(best_ids.squeeze(0)), best
        return res
