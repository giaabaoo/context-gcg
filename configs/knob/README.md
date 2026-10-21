# configs/knob/

Rank-1 context/prior directions from Minder et al., one per model, used by the steering baseline
(`rq1_baselines.py --method steering`).

| file | source | layer | d |
|---|---|---|---|
| `llama.pt` | `jkminder/CTXPRIOR-Projection-Meta-Llama-3.1-8B-Instruct-L16` | 16 | 4096 |
| `mistral.pt` | `jkminder/CTXPRIOR-Projection-Mistral-7B-Instruct-v0.3-L16` | 16 | 4096 |
| `gemma.pt` | `jkminder/CTXPRIOR-Projection-gemma-2-9b-it-L27` | 27 | 3584 |

Each file holds `{"u", "layer", "repo", "value_prior", "value_context"}`, where `u` is the unit
vector read out of their projection. Delete one and `contextgcg/baselines/steering.py` downloads
the projection and rebuilds it. The layers and clamp values are in `contextgcg/models.py`.

```bibtex
@inproceedings{minder2025controllable,
  title     = {Controllable Context Sensitivity and the Knob Behind It},
  author    = {Minder, Julian and Du, Kevin and Stoehr, Niklas and Monea, Giovanni and
               Wendler, Chris and West, Robert and Cotterell, Ryan},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2025},
  eprint    = {2411.07404},
  archivePrefix = {arXiv},
  url       = {https://arxiv.org/abs/2411.07404},
}
```
