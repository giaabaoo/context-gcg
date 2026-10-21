# data/

```
items/<set>/<model>.jsonl                  inputs
runs/<section>/<model>/<condition>.jsonl   outputs, one line per answer
runs/<section>/counts.json                 totals for cells with no per-answer labels
rubrics/*.txt                              the scorers
example_passage.yml                        a passage to copy for run_suffix.py
```

## Fields

| | |
|---|---|
| `label` | the label the paper uses |
| `labels.human` | human annotator labels. 7,237 answers |
| `labels.claude` | Claude Opus 5 judge labels. 23,981 answers |
| `labels.qwen` | Qwen3-32B judge labels, under `rubrics/flip.txt`. 11,020 answers |
| `label: null` | not scored on its own; the rate is in `counts.json` |
| `src` | where the answer came from in our research repo |

Each line is keyed by `item_id`, plus `turn` or `question_id`.

The human annotator and Claude Opus 5 agree at kappa 0.990 on the 1,459 answers carrying both.

About 21 labels are wrong, in the conditions that answer in paragraphs such as `cad`. Correcting
them lowers those baselines, not the suffix.

`label` is usually `true` or `false`. The swap condition uses `new`, `adopted` or `neither`, and
ten Gemma answers use `echoes_suffix`, where the model repeated the suffix instead of answering.
