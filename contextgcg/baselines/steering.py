from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

import torch
from torch import nn


def load_direction(repo: str, cache: str | Path | None = None) -> torch.Tensor:
    if cache is not None and Path(cache).exists():
        u = torch.load(cache, weights_only=True)["u"].float()
    else:
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file

        class _Proj(nn.Module):
            def __init__(self, d: int):
                super().__init__()
                self.weight = nn.Parameter(torch.empty(d, 1))
                nn.init.orthogonal_(self.weight)
                nn.utils.parametrizations.orthogonal(self)

        sd = load_file(hf_hub_download(repo, "model.safetensors"))
        m = _Proj(sd["parametrizations.weight.original"].shape[0])
        m.load_state_dict(sd, strict=False)
        u = m.weight.detach().squeeze(-1).contiguous().float()
    assert abs(u.norm().item() - 1.0) < 1e-4, "knob direction must be unit norm"
    return u


@contextmanager
def clamp(model, u: torch.Tensor, layer: int, value: float):
    u = (u / u.norm()).to(model.device, model.dtype)

    def hook(module, args, output):
        h = output[0]
        uu = u.to(h)
        last = h[:, -1, :]
        h[:, -1, :] = last - (last @ uu).unsqueeze(-1) * uu + value * uu
        return (h, *output[1:])

    handle = model.model.layers[layer].register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()
