from __future__ import annotations

import torch


def init_suffix(suffix_len: int = 20, char: str = "x") -> str:
    return " ".join([char] * suffix_len)


def random_suffix(tok, index: int, suffix_len: int = 20, seed: int = 0) -> str:
    gen = torch.Generator().manual_seed(seed + index)
    ids = torch.randint(0, tok.vocab_size, (suffix_len,), generator=gen).tolist()
    return tok.decode(ids, skip_special_tokens=True)
