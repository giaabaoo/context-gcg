from __future__ import annotations

import torch

from contextgcg.data import Item
from contextgcg.prompting import no_context


def prompts(tok, item: Item, chat: bool) -> tuple[str, str]:
    if not chat:
        return (f"Context: {item.passage}\nQuestion: {item.query}\nAnswer:",
                f"Question: {item.query}\nAnswer:")

    def wrap(body: str) -> str:
        t = tok.apply_chat_template([{"role": "user", "content": body}], tokenize=False,
                                    add_generation_prompt=True)
        return t.replace(tok.bos_token, "", 1) if tok.bos_token and t.startswith(tok.bos_token) else t
    return wrap(f"Context: {item.passage}\nQuestion: {item.query}"), wrap(no_context(item.query))


class CAD:
    def __init__(self, model, tok, max_new_tokens: int = 256) -> None:
        self.model, self.tok, self.n = model, tok, max_new_tokens
        self.dev = model.device
        self.hybrid = getattr(model.config, "model_type", "") == "gemma2"

    def _prime(self, enc):
        n = enc["input_ids"].shape[1]
        if not self.hybrid:
            out = self.model(**enc, use_cache=True)
            return out.logits[:, -1, :].float(), out.past_key_values, n
        from transformers.cache_utils import HybridCache
        cache = HybridCache(config=self.model.config, batch_size=1, max_cache_len=n + self.n + 1,
                            device=self.dev, dtype=self.model.dtype)
        out = self.model(**enc, past_key_values=cache, use_cache=True,
                         cache_position=torch.arange(n, device=self.dev))
        return out.logits[:, -1, :].float(), cache, n

    def _step(self, tok_id: int, cache, pos: int):
        t = torch.tensor([[tok_id]], device=self.dev)
        if not self.hybrid:
            out = self.model(input_ids=t, past_key_values=cache, use_cache=True)
            return out.logits[:, -1, :].float(), out.past_key_values
        out = self.model(input_ids=t, past_key_values=cache, use_cache=True,
                         cache_position=torch.tensor([pos], device=self.dev))
        return out.logits[:, -1, :].float(), cache

    @torch.no_grad()
    def __call__(self, item: Item, alpha: float, chat: bool = True) -> str:
        p_ctx, p_noctx = prompts(self.tok, item, chat)
        la, ka, na = self._prime(self.tok(p_ctx, return_tensors="pt").to(self.dev))
        lb, kb, nb = self._prime(self.tok(p_noctx, return_tensors="pt").to(self.dev))
        stop = {self.tok.eos_token_id}
        if self.hybrid:
            stop.add(self.tok.convert_tokens_to_ids("<end_of_turn>"))
        ids: list[int] = []
        for step in range(self.n):
            nxt = int(torch.argmax((1.0 + alpha) * la - alpha * lb, dim=-1))
            if nxt in stop:
                break
            ids.append(nxt)
            la, ka = self._step(nxt, ka, na + step)
            lb, kb = self._step(nxt, kb, nb + step)
        return self.tok.decode(ids, skip_special_tokens=True).strip()
