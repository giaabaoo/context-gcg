from __future__ import annotations

from contextlib import nullcontext

import torch

from contextgcg.prompting import render


class Generator:
    def __init__(self, model, tok, max_new_tokens: int = 256, batch_size: int = 8,
                 stop_at_eot: str | None = None) -> None:
        self.model, self.tok = model, tok
        self.max_new_tokens, self.batch_size = max_new_tokens, batch_size
        self.eos = tok.convert_tokens_to_ids(stop_at_eot) if stop_at_eot else None

    @torch.no_grad()
    def __call__(self, conversations: list[list[dict]], hooks=None) -> list[str]:
        tok = self.tok
        was = tok.padding_side
        tok.padding_side = "left"
        try:
            out = self._run(conversations, hooks)
        finally:
            tok.padding_side = was
        return out

    def _run(self, conversations: list[list[dict]], hooks) -> list[str]:
        tok = self.tok
        out: list[str] = []
        for i in range(0, len(conversations), self.batch_size):
            texts = [render(tok, c) for c in conversations[i:i + self.batch_size]]
            enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False)
            enc = enc.to(self.model.device)
            extra = {"eos_token_id": self.eos} if self.eos is not None else {}
            with hooks() if hooks is not None else nullcontext():
                gen = self.model.generate(**enc, max_new_tokens=self.max_new_tokens, do_sample=False,
                                          pad_token_id=tok.pad_token_id, **extra)
            n = enc["input_ids"].shape[1]
            out += [tok.decode(g[n:], skip_special_tokens=True).strip() for g in gen]
        return out
