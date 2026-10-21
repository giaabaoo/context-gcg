"""The question asked again, with no new evidence.

    python experiments/rq5_persistence.py --model llama --condition reask
    python experiments/rq5_persistence.py --model llama --condition repair

reask runs turns 2 and 3 after a stored turn 1. repair re-optimizes one suffix per item: joint
fits turn 1 and the re-ask together, control fits turn 1 only.

Writes out/rq5/<model>/<condition>.jsonl, one row per turn.
"""
from __future__ import annotations

import _common as C
from contextgcg.experiments import multiturn
from contextgcg.prompting import doc, reask as reask_msgs, user

REPAIR_TARGET = {"llama": "{ck}.", "mistral": "{ck}", "gemma": "{ck}.<end_of_turn>"}


def main() -> None:
    p = C.parser(__doc__)
    p.add_argument("--condition", required=True, choices=["reask", "repair"])
    p.add_argument("--steps", type=int, default=150,
                   help="repair search steps (paper: 150)")
    p.add_argument("--instruction", action="store_true",
                   help="reask only: put the instruction baseline in the slot instead of a suffix")
    a = p.parse_args()
    S = C.Session(a)
    pool = {it.id: it for it in C.items("persist" if a.condition == "reask" else "repair", a.model)}

    if a.condition == "reask":
        src = "instruction" if a.instruction else "suffix"
        turn1 = [r for r in C.released("persist", a.model, src) if int(r["turn"]) == 1]
        turn1 = C.clip([r for r in turn1 if r["item_id"] in pool], a.limit)
        slot_of = lambda r: r.get("suffix") or r.get("instruction") or ""
        convs = []
        for r in turn1:
            it = pool[r["item_id"]]
            convs.append(reask_msgs(doc(it.passage, it.query, slot_of(r)), r["answer"], it.query))
        if a.dry_run:
            C.show_prompt(S.tok, convs)
            print(f"[dry-run] {len(convs)} conversations; turn 3 is built from each turn-2 answer.")
            rows = []
        else:
            t2 = S.generate(convs, batch_size=16)
            t3 = S.generate([c + [{"role": "assistant", "content": x},
                                  {"role": "user", "content": pool[r["item_id"]].query}]
                             for c, x, r in zip(convs, t2, turn1)], batch_size=16)
            rows = []
            for r, x2, x3 in zip(turn1, t2, t3):
                rows += [{"item_id": r["item_id"], "turn": 1, "answer": r["answer"], "slot": slot_of(r)},
                         {"item_id": r["item_id"], "turn": 2, "answer": x2},
                         {"item_id": r["item_id"], "turn": 3, "answer": x3}]
            print(f"turn 1 answers are the released ones, so turns 2 and 3 start from the same place.")
    else:
        rows = []
        for condition in ("control", "joint"):
            ids = list(dict.fromkeys(r["item_id"] for r in C.released("repair", a.model, condition)))
            ids = C.clip([i for i in ids if i in pool], a.limit)
            print(f"== {condition}: {len(ids)} items (these held turn 1 and then lost turn 2)")
            if a.dry_run:
                print(f"   would re-search {a.steps} steps per item, target {REPAIR_TARGET[a.model]!r}, "
                      f"then regenerate turns 1-3.")
                continue
            model, tok = S.model_tok
            for i, iid in enumerate(ids, 1):
                r = multiturn.repair(a.model, pool[iid], REPAIR_TARGET[a.model], condition,
                                     steps=a.steps, seed=1, model=model, tok=tok)
                print(f"   {i}/{len(ids)} {iid}")
                rows += [{"item_id": iid, "condition": condition, "turn": t, "suffix": r["suffix"],
                          "answer": r[f"turn{t}"]} for t in (1, 2, 3)]

    if rows and not a.dry_run:
        name = f"{a.condition}_{'instruction' if a.instruction else 'suffix'}" if a.condition == "reask" else a.condition
        C.save(C.out_dir(a, "rq5") / f"{name}.jsonl", rows)


if __name__ == "__main__":
    main()
