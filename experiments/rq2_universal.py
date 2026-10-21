"""One suffix for many passages.

    python experiments/rq2_universal.py --model mistral
    python experiments/rq2_universal.py --model mistral --nonprogressive
    python experiments/rq2_universal.py --model mistral --released

Fitted on training passages, then applied unchanged to every passage in the pool. progressive
adds goals as the current ones are solved; nonprogressive uses a fixed 20.

Writes out/rq2/<model>/<schedule>.jsonl, with split = train or unseen.
"""
from __future__ import annotations

import json

import _common as C
from contextgcg.experiments import universal
from contextgcg.optim.universal import UniversalConfig
from contextgcg.prompting import doc, user

TARGET = {"llama": "{ck}", "mistral": "{ck}.", "gemma": " {ck}.<end_of_turn>"}


def main() -> None:
    p = C.parser(__doc__)
    p.add_argument("--nonprogressive", action="store_true",
                   help="fit on a fixed 20 passages instead of the progressive schedule")
    p.add_argument("--released", action="store_true", help="use our suffix; skip the search")
    p.add_argument("--steps", type=int, default=500,
                   help="search steps (paper: 500)")
    a = p.parse_args()
    schedule = "nonprogressive" if a.nonprogressive else "progressive"
    S = C.Session(a)

    splits = json.loads((C.DATA / "items" / "universal_train" / "splits.json").read_text())[a.model]
    pool = C.items("basefakepedia", a.model)
    train_ids = set(splits["train20"] if a.nonprogressive else splits["progressive_train_ids"])

    if a.released:
        key = "nonprog" if a.nonprogressive else "progressive"
        suffix = json.loads((C.DATA / "items" / "universal_train" / "suffixes.json").read_text())[a.model][key]["suffix"]
        print(f"our {schedule} suffix for {a.model}:\n  {suffix!r}")
    elif a.dry_run:
        suffix = " ".join(["x"] * 20)
        n = len(train_ids)
        print(f"[dry-run] would fit one suffix on {n} training passages ({schedule}, {a.steps} steps), "
              f"then answer all {len(pool)} passages with it.")
    else:
        if a.nonprogressive:
            by = {it.id: it for it in pool}
        else:
            by = {it.id: it for it in C.items("universal_train", a.model)}
        train = [by[i] for i in (splits["train20"] if a.nonprogressive else splits["progressive_train_ids"])
                 if i in by]
        cfg = UniversalConfig(target=TARGET[a.model], steps=a.steps, progressive=not a.nonprogressive)
        model, tok = S.model_tok
        res = universal.search(a.model, train, cfg, C.out_dir(a, "rq2") / schedule, model=model, tok=tok)
        suffix = res["suffix"]
        print(f"fitted suffix:\n  {suffix!r}\n  selection loss {res['select_loss']:.4f}")

    evaluated = C.clip(pool, a.limit)
    answers = S.generate([user(doc(it.passage, it.query, suffix)) for it in evaluated], batch_size=16)
    rows = [{"item_id": it.id, "split": "train" if it.id in train_ids else "unseen",
             "suffix": suffix, "answer": ans} for it, ans in zip(evaluated, answers)]
    if not a.dry_run:
        C.save(C.out_dir(a, "rq2") / f"{schedule}.jsonl", rows)
        n_unseen = sum(r["split"] == "unseen" for r in rows)
        print(f"{n_unseen} of {len(rows)} passages were never in the search's training set.")
    C.judge(a, rows, {it.id: it for it in evaluated})
    print("`split` in the output marks the unseen passages.")


if __name__ == "__main__":
    main()
