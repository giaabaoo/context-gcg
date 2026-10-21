"""Passage controls. All three reuse the per-item suffixes from RQ1 and change the passage.

    python experiments/rq3_passage_controls.py --model llama --control carry
    python experiments/rq3_passage_controls.py --model llama --control swap
    python experiments/rq3_passage_controls.py --model llama --control domain

carry weakens the passage: unchanged, planted answer masked, planted answer substituted, passage
deleted, each also without the suffix. swap gives the suffix a different item's passage. domain
uses passages outside conflict.

Writes out/rq3/<model>/<control>_<condition>.jsonl.
"""
from __future__ import annotations

import random

import _common as C
from contextgcg.prompting import doc_tight, passage_deleted, user

DOMAIN_SETS = {"true": "domain_true", "noprior": "domain_noprior", "new": "domain_new"}


def suffixes_for(model: str, ids: set[str]) -> dict[str, str]:
    return {r["item_id"]: r["suffix"] for r in C.released("q1", model, "suffix_seed0")
            if r["item_id"] in ids}


def main() -> None:
    p = C.parser(__doc__)
    p.add_argument("--control", required=True, choices=["carry", "swap", "domain"])
    a = p.parse_args()
    S = C.Session(a)
    out = C.out_dir(a, "rq3")

    if a.control == "domain":
        conditions = {}
        for name, dataset in DOMAIN_SETS.items():
            pool = C.clip(C.items(dataset, a.model), a.limit)
            shared = {r["suffix"] for r in C.released("domain", a.model, f"{name}_suffix")
                      if r.get("suffix")}
            if len(shared) != 1:
                raise SystemExit(f"domain/{name}: expected one shared suffix, found {len(shared)}")
            sfx = shared.pop()
            conditions[f"{name}_nosuffix"] = (pool, [""] * len(pool))
            conditions[f"{name}_suffix"] = (pool, [sfx] * len(pool))
        by_condition = {}
        for condition, (pool, fills) in conditions.items():
            if not pool:
                continue
            print(f"== {condition}: {len(pool)} passages")
            ans = S.generate([user(doc_tight(it.passage, it.query, f)) for it, f in zip(pool, fills)],
                             batch_size=8, max_new_tokens=192 if condition.startswith(("new", "noprior")) else 256)
            by_condition[condition] = [{"item_id": it.id, "answer": x} for it, x in zip(pool, ans)]
            if a.dry_run:
                break

    elif a.control == "carry":
        pool = C.clip(C.items("mask", a.model), a.limit)
        sfx = suffixes_for(a.model, {it.id for it in pool})
        pool = [it for it in pool if it.id in sfx]
        variants = {"unchanged": lambda it: it.passage,
                    "masked": lambda it: it.meta["passage_masked"],
                    "substituted": lambda it: it.meta["passage_substituted"]}
        by_condition = {}
        for name, get in variants.items():
            for tag, fill in (("", True), ("_nosuffix", False)):
                if name == "unchanged" and not fill:
                    continue
                print(f"== {name}{tag}")
                convs = [user(doc_tight(get(it), it.query, sfx[it.id] if fill else "")) for it in pool]
                ans = S.generate(convs, batch_size=16)
                by_condition[f"{name}{tag}"] = [{"item_id": it.id, "answer": x}
                                               for it, x in zip(pool, ans)]
                if a.dry_run:
                    break
            if a.dry_run:
                break
        if not a.dry_run:
            print("== deleted (passage removed, suffix kept)")
            print("The shipped deleted answers came from an earlier run, with other suffixes and a\n"
                  "48-token decode, so these will not match them.")
            ans = S.generate([user(passage_deleted(sfx[it.id], it.query)) for it in pool], batch_size=16)
            by_condition["deleted"] = [{"item_id": it.id, "answer": x} for it, x in zip(pool, ans)]

    else:
        pool = C.clip(C.items("basefakepedia", a.model), a.limit)
        sfx = suffixes_for(a.model, {it.id for it in pool})
        pool = [it for it in pool if it.id in sfx]
        order = list(range(len(pool)))
        random.Random(0).shuffle(order)
        pairs = [(it, pool[j]) for it, j in zip(pool, order) if pool[j].id != it.id]
        if not pairs:
            raise SystemExit(f"swap needs at least two items that pair: got {len(pool)}. "
                             f"Raise --limit.")
        print(f"== swap: {len(pairs)} pairs (own question + suffix, another item's passage)")
        ans = S.generate([user(doc_tight(other.passage, it.query, sfx[it.id])) for it, other in pairs],
                         batch_size=16)
        by_condition = {"swap": [{"item_id": it.id, "swapped_in_item_id": other.id,
                                 "own_planted": it.planted, "new_planted": other.planted, "answer": x}
                                for (it, other), x in zip(pairs, ans)]}
        print("Read each answer as: the new passage's fact, the item's own planted fact, or neither.")

    if not a.dry_run:
        for condition, rows in by_condition.items():
            C.save(out / f"{a.control}_{condition}.jsonl", rows)
    if a.control == "swap":
        print("The shipped swap and deletion answers came from an earlier run, with other suffixes\n"
              "and a 48-token decode; this script uses the RQ1 suffixes at 256, so expect close but\n"
              "not identical rates.")


if __name__ == "__main__":
    main()
