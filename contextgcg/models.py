from __future__ import annotations

from dataclasses import dataclass

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

_DTYPE_KW = "dtype" if int(transformers.__version__.split(".")[0]) >= 5 else "torch_dtype"


@dataclass(frozen=True)
class Knob:
    repo: str
    layer: int
    context_value: float


@dataclass(frozen=True)
class ModelSpec:
    name: str
    repo: str
    eot: str
    prefix_cache: bool
    knob: Knob
    dtype: str = "bfloat16"


MODELS = {
    "llama": ModelSpec("llama", "NousResearch/Meta-Llama-3.1-8B-Instruct", "<|eot_id|>", True,
                       Knob("jkminder/CTXPRIOR-Projection-Meta-Llama-3.1-8B-Instruct-L16", 16, -6.0)),
    "mistral": ModelSpec("mistral", "mistralai/Mistral-7B-Instruct-v0.3", "</s>", True,
                         Knob("jkminder/CTXPRIOR-Projection-Mistral-7B-Instruct-v0.3-L16", 16, -5.0)),
    "gemma": ModelSpec("gemma", "unsloth/gemma-2-9b-it", "<end_of_turn>", False,
                       Knob("jkminder/CTXPRIOR-Projection-gemma-2-9b-it-L27", 27, 150.0)),
}


def load_tokenizer(spec: ModelSpec):
    tok = AutoTokenizer.from_pretrained(spec.repo)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return tok


def load(spec: ModelSpec, device: str | None = None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[spec.dtype]
    model = AutoModelForCausalLM.from_pretrained(spec.repo, **{_DTYPE_KW: dtype}).to(device).eval()
    return model, load_tokenizer(spec)
