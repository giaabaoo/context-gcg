from __future__ import annotations

from dataclasses import dataclass

import nanogcg
from nanogcg import GCGConfig

from contextgcg.data import Item
from contextgcg.prompting import GCG_SLOT, doc


@dataclass(frozen=True)
class SearchConfig:
    steps: int = 100
    topk: int = 256
    search_width: int = 512
    suffix_len: int = 20
    init_char: str = "x"
    target: str = "{ck}.{eot}"
    seed: int = 0

    @property
    def init(self) -> str:
        return " ".join([self.init_char] * self.suffix_len)


@dataclass(frozen=True)
class SearchResult:
    suffix: str
    loss: float
    losses: list


class SuffixSearch:
    def __init__(self, model, tok, eot: str, prefix_cache: bool, cfg: SearchConfig) -> None:
        self.model, self.tok, self.eot, self.cfg = model, tok, eot, cfg
        self.prefix_cache = prefix_cache

    def gcg_config(self) -> GCGConfig:
        c = self.cfg
        return GCGConfig(num_steps=c.steps, optim_str_init=c.init, search_width=c.search_width,
                         topk=c.topk, early_stop=False, use_prefix_cache=self.prefix_cache,
                         add_space_before_target=False, seed=c.seed, verbosity="WARNING")

    def message(self, item: Item) -> str:
        return doc(item.passage, item.query, GCG_SLOT)

    def target(self, item: Item) -> str:
        return self.cfg.target.format(ck=item.planted, eot=self.eot)

    def __call__(self, item: Item) -> SearchResult:
        r = nanogcg.run(self.model, self.tok, self.message(item), self.target(item), self.gcg_config())
        return SearchResult(r.best_string, float(r.best_loss), [float(x) for x in r.losses])
