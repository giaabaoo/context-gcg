"""One suffix per item.

    python experiments/rq1_per_item.py --model llama
    python experiments/rq1_per_item.py --model llama --released
    python experiments/rq1_per_item.py --model gemma --items cubnq

100 steps, 20 tokens, target "{planted}.{eot}", suffix at the end of the passage, then a
256-token answer. Writes out/rq1/<model>/<items>_seed<S>.jsonl.
"""
from __future__ import annotations

import _common as C
from contextgcg.experiments import per_item
from contextgcg.optim.gcg import SearchConfig, SuffixSearch
from contextgcg.prompting import doc, user

BEST_SEED = {"llama": 2, "mistral": 1, "gemma": 1}


def main() -> None:
    p = C.parser(__doc__)
    p.add_argument("--items", default="basefakepedia", choices=["basefakepedia", "cubnq"])
    p.add_argument("--seed", type=int, default=None, help="default: BEST_SEED for this model")
    p.add_argument("--released", action="store_true", help="use our suffixes; skip the search")
    a = p.parse_args()
    seed = a.seed if a.seed is not None else (0 if a.items == "cubnq" else BEST_SEED[a.model])
    S = C.Session(a)
    pool = C.clip(C.items(a.items, a.model), a.limit)
    by_id = {it.id: it for it in pool}
    tag = "released_" if a.released else ""
    out = C.out_dir(a, "rq1") / f"{tag}{a.items}_seed{seed}.jsonl"

    if a.released:
        condition = ("q1", f"suffix_seed{seed}") if a.items == "basefakepedia" else ("nq", "ours")
        rows = [{"item_id": r["item_id"], "suffix": r["suffix"]}
                for r in C.released(condition[0], a.model, condition[1]) if r["item_id"] in by_id]
    elif a.dry_run:
        it = pool[0]
        s = SuffixSearch(None, S.tok, S.spec.eot, S.spec.prefix_cache, SearchConfig(seed=seed))
        print(f"[dry-run] would search {len(pool)} items. Item 1: the optimised message "
              f"(nanoGCG fills {{optim_str}}) and the target:\n  {s.message(it)!r}\n  -> {s.target(it)!r}")
        rows = [{"item_id": it.id, "suffix": " ".join(["x"] * 20)} for it in pool]
    else:
        search_out = out.with_name(out.stem + ".suffixes.jsonl")
        rows = per_item.search(a.model, pool, search_out, seed=seed,
                               model=S.model, tok=S.model_tok[1])

    answers = S.generate([user(doc(by_id[r["item_id"]].passage, by_id[r["item_id"]].query, r["suffix"]))
                          for r in rows], batch_size=8)
    rows = [{**r, "answer": ans} for r, ans in zip(rows, answers)]
    if not a.dry_run:
        C.save(out, rows)
    C.judge(a, rows, by_id)


if __name__ == "__main__":
    main()
