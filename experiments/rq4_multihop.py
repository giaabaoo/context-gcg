"""Held-out questions about the planted fact.

    python experiments/rq4_multihop.py --model llama
    python experiments/rq4_multihop.py --model llama --single
    python experiments/rq4_multihop.py --model llama --released

Each passage gets a suffix fitted on the base question plus K multi-hop questions; the rest are
held out and scored. --single spends the same budget on the base question alone.

Writes out/rq4/<model>/{multi,single}.jsonl.
"""
from __future__ import annotations

import _common as C
from contextgcg.experiments import multihop
from contextgcg.prompting import doc, user
from contextgcg.io import read_jsonl

def bank(model: str) -> list[dict]:
    out = []
    for r in read_jsonl(C.DATA / "items" / "entailed" / f"{model}.jsonl"):
        qs = r["meta"]["questions"]
        out.append({"id": r["id"], "passage": r["passage"], "base_question": r["query"],
                    "planted": r["planted"],
                    "train": [q for q in qs if q["split"] == "train"],
                    "test": [q for q in qs if q["split"] != "train"]})
    return out


def main() -> None:
    p = C.parser(__doc__)
    p.add_argument("--single", action="store_true", help="one-goal control: base question only")
    p.add_argument("--released", action="store_true", help="use our suffixes; skip the search")
    p.add_argument("--steps", type=int, default=250,
                   help="search steps (paper: 250)")
    a = p.parse_args()
    condition = "single" if a.single else "multi"
    S = C.Session(a)
    entries = C.clip(bank(a.model), a.limit)

    if a.released or a.dry_run:
        stored = {r["item_id"]: r["suffix"] for r in C.released("entailed", a.model, condition)} \
            if a.released else {}
        rows = []
        for e in entries:
            sfx = stored.get(e["id"], " ".join(["x"] * 20))
            rows += [{"item_id": e["id"], "question_id": t["question_id"], "question": t["question"],
                      "target": t.get("target", ""), "suffix": sfx} for t in e["test"]]
        if a.dry_run:
            g, w = multihop.goals(entries[0], S.spec.eot, a.single)
            print(f"[dry-run] {condition}: {len(g)} goal(s) per passage, weights {w}.")
            print(f"  goal 1 target: {g[0][1]!r}")
            if len(g) > 1:
                print(f"  goal 2 target: {g[1][1]!r}")
            print(f"  {len(rows)} held-out questions would be answered.")
        answers = S.generate([user(doc(next(e for e in entries if e["id"] == r["item_id"])["passage"],
                                       r["question"], r["suffix"])) for r in rows], batch_size=1)
        rows = [{**r, "answer": ans} for r, ans in zip(rows, answers)]
    else:
        model, tok = S.model_tok
        rows = []
        for i, e in enumerate(entries, 1):
            print(f"== passage {i}/{len(entries)}: {e['id']}")
            res = multihop.run(a.model, e, single=a.single, model=model, tok=tok,
                               steps=a.steps)
            rows += [{"item_id": e["id"], "question": t["question"], "suffix": res["suffix"],
                      "loss": res["loss"], "answer": t["answer"]} for t in res["test"]]

    if not a.dry_run:
        tag = "released_" if a.released else ""
        C.save(C.out_dir(a, "rq4") / f"{tag}{condition}.jsonl", rows)
    print("Held-out questions only; --judge scores them with the multi-hop rubric.")


if __name__ == "__main__":
    main()
