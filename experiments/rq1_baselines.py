"""Baselines, on the same items as the suffix.

    python experiments/rq1_baselines.py --model mistral --method instructions
    python experiments/rq1_baselines.py --model llama --method cad --items cubnq

methods: instructions (ten phrasings), controls (the search init string and a random suffix),
cad (Shi et al. 2024), steering (Minder et al.).

Writes out/rq1_baselines/<model>/<items>_<condition>.jsonl.
"""
from __future__ import annotations

import _common as C
from contextgcg import prompting as P
from contextgcg.baselines.controls import init_suffix, random_suffix

def run_slots(S, a, pool, slots: dict[str, list[str]], batch_size: int) -> dict[str, list[dict]]:
    conditions = {}
    for name, fills in slots.items():
        print(f"== {name}")
        ans = S.generate([P.user(P.doc(it.passage, it.query, f)) for it, f in zip(pool, fills)],
                         batch_size=batch_size)
        conditions[name] = [{"item_id": it.id, "slot": f, "answer": x} for it, f, x in zip(pool, fills, ans)]
        if a.dry_run:
            break
    return conditions


def main() -> None:
    p = C.parser(__doc__)
    p.add_argument("--method", required=True,
                   choices=["instructions", "controls", "cad", "steering"])
    p.add_argument("--items", default="basefakepedia", choices=["basefakepedia", "cubnq"])
    a = p.parse_args()
    S = C.Session(a)
    pool = C.clip(C.items(a.items, a.model), a.limit)
    by_id = {it.id: it for it in pool}

    if a.method == "instructions":
        formats = [True] if a.items == "basefakepedia" else [False, True]
        slots = {f"instr_{'oneword' if ow else 'free'}_{k}": [P.instruction(k, ow)] * len(pool)
                 for ow in formats for k in P.INSTRUCTIONS}
        conditions = run_slots(S, a, pool, slots, batch_size=16)
    elif a.method == "controls":
        tok = S.tok
        slots = {"init": [init_suffix()] * len(pool),
                 "random": [random_suffix(tok, i) for i in range(len(pool))]}
        conditions = run_slots(S, a, pool, slots, batch_size=8)
    elif a.method == "cad":
        from contextgcg.baselines.cad import CAD, prompts
        if a.dry_run:
            print("[dry-run] CAD runs two streams; with-passage prompt:\n" + prompts(S.tok, pool[0], True)[0])
            conditions = {}
        else:
            model, tok = S.model_tok
            cad = CAD(model, tok, 256)
            conditions = {"cad": [{"item_id": it.id, "answer": cad(it, 2.0, True)} for it in pool]}
    else:
        from contextgcg.baselines.steering import clamp, load_direction
        convs = [P.minder(it.passage, it.query) for it in pool]
        conditions = {"minder_prompt": S.generate(convs, batch_size=1)}
        if not a.dry_run:
            k = S.spec.knob
            u = load_direction(k.repo, C.ROOT / "configs" / "knob" / f"{a.model}.pt")
            conditions["minder_steer"] = S.generate(
                convs, batch_size=1, hooks=lambda: clamp(S.model, u, k.layer, k.context_value))
        conditions = {n: [{"item_id": it.id, "answer": x} for it, x in zip(pool, xs)] for n, xs in conditions.items()}

    if not a.dry_run:
        d = C.out_dir(a, "rq1_baselines")
        for name, rows in conditions.items():
            C.save(d / f"{a.items}_{name}.jsonl", rows)
            C.judge(a, rows, by_id)


if __name__ == "__main__":
    main()
