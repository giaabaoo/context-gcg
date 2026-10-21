from __future__ import annotations

from pathlib import Path

import torch

RUBRICS = Path(__file__).resolve().parent.parent / "data" / "rubrics"
JUDGE_REPO = "Qwen/Qwen3-32B"


def load_rubric(name: str) -> str:
    raw = (RUBRICS / f"{name}.txt").read_text()
    text = "\n".join(l for l in raw.split("\n") if not l.startswith("#"))
    return text.strip() if name == "flip" else text


class QwenJudge:
    def __init__(self, model, tok, rubric: str) -> None:
        self.model, self.tok, self.name = model, tok, rubric
        self.rubric = load_rubric(rubric)
        self.a, self.b = (tok.encode(x, add_special_tokens=False)[0] for x in ("A", "B"))

    def prompt(self, *, passage: str, query: str, planted: str, true: str, answer: str,
               claim: str = "") -> str:
        r = self.rubric
        if self.name == "flip":
            body = r.format(context=passage, query=query, ck=planted, pk=true, text=answer)
        elif self.name.startswith("multihop"):
            body = (r.replace("{claim}", claim).replace("{context}", claim).replace("{query}", query)
                    .replace("{reference}", planted).replace("{ck}", planted).replace("{text}", answer))
        else:
            body = (r.replace("{context}", passage).replace("{query}", query).replace("{ck}", planted)
                    .replace("{pk}", str(true or "")).replace("{text}", answer))
        return self.tok.apply_chat_template([{"role": "user", "content": body}], tokenize=False,
                                            add_generation_prompt=True, enable_thinking=False)

    @torch.no_grad()
    def __call__(self, **fields) -> bool:
        ids = self.tok(self.prompt(**fields), return_tensors="pt",
                       add_special_tokens=False).input_ids.to(self.model.device)
        lg = self.model(ids).logits[0, -1]
        if self.name == "flip":
            p = torch.softmax(lg[[self.a, self.b]].float(), dim=-1)
            return bool(p[0] >= p[1])
        return bool(lg[self.a] > lg[self.b])
