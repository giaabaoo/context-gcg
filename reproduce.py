"""Re-run a released condition.

    python reproduce.py conditions          list every condition you can address
    python reproduce.py compare A B         compare two conditions' rates (no GPU)
    python reproduce.py read COND           prompts for scoring an entailed condition by reading
    python reproduce.py answers COND        re-generate one condition's answers (1 GPU)
    python reproduce.py judge   COND        re-label one condition with Qwen3-32B (1 GPU)
    python reproduce.py search  EXPERIMENT  re-run a search (many GPU hours)

COND is section/model/condition, e.g. q1/llama/suffix_seed2. Each condition has a recipe in
configs/conditions/ giving the layout, decoding settings and rubric of the original run.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from contextgcg.reading import ReadingError
from contextgcg.replay import ReplayError

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
MODELS = ("llama", "mistral", "gemma")


def cmd_conditions(a) -> int:
    for sec in sorted(p.name for p in (DATA / "runs").iterdir() if p.is_dir()):
        for m in MODELS:
            d = DATA / "runs" / sec / m
            if d.is_dir():
                print(f"{sec}/{m}: {' '.join(sorted(p.stem for p in d.glob('*.jsonl')))}")
    return 0


def _labelled(condition: str) -> dict:
    from contextgcg.io import read_jsonl
    section, model, name = condition.split("/")
    path = DATA / "runs" / section / model / f"{name}.jsonl"
    if not path.exists():
        raise SystemExit(f"no condition {condition} (see `python reproduce.py conditions`)")
    out = {}
    for r in read_jsonl(path):
        if not isinstance(r.get("label"), bool):
            continue
        key = (r["item_id"], r.get("question_id"), r.get("turn"))
        out[key] = r["label"]
    if not out:
        raise SystemExit(f"{condition}: no answer in it carries a label; only its total was kept, "
                         f"in data/runs/<section>/counts.json")
    return out


def cmd_compare(a) -> int:
    from contextgcg.stats import mcnemar_exact, paired_discordant, passage_bootstrap
    A, B = _labelled(a.condition_a), _labelled(a.condition_b)
    won, lost, n = paired_discordant(A, B)
    if not n:
        raise SystemExit("the two conditions share no item; compare conditions of the same section and model")
    ra = sum(A[k] for k in A) / len(A)
    rb = sum(B[k] for k in B) / len(B)
    pa = sum(A[k] for k in sorted(set(A) & set(B))) / n
    pb = sum(B[k] for k in sorted(set(A) & set(B))) / n
    print(f"{a.condition_a}: {ra:.3f} over {len(A)} labelled answers")
    print(f"{a.condition_b}: {rb:.3f} over {len(B)} labelled answers")
    print(f"paired on the {n} items both carry: {pa:.3f} vs {pb:.3f}")
    print(f"exact McNemar on {won + lost} discordant pairs ({won} / {lost}): p = "
          f"{mcnemar_exact(won, lost):.4g}")
    if a.bootstrap:
        x = {k: int(A[k]) for k in sorted(set(A) & set(B))}
        y = {k: int(B[k]) for k in sorted(set(A) & set(B))}
        r = passage_bootstrap(x, y)
        print(f"passage bootstrap ({r['passages']} passages, {r['n']} answers): "
              f"diff {r['diff']:+.3f}, 95% CI [{r['lo']:+.3f}, {r['hi']:+.3f}], "
              f"{'p < ' + format(2 / r['reps'], '.1e') if r['p'] == 0 else 'p = ' + format(r['p'], '.4g')}")
    return 0


def cmd_read(a) -> int:
    from contextgcg.judge import load_rubric
    from contextgcg.reading import from_condition, from_sample, prompts, verdicts
    from contextgcg.stats import cohen_kappa
    rows = from_sample(DATA) if a.condition == "sample" else from_condition(DATA, a.condition)
    if a.limit:
        rows = rows[:a.limit]
    if not a.score:
        for row in prompts(load_rubric("multihop_reading"), rows):
            print(json.dumps(row, ensure_ascii=False))
        print(f"{len(rows)} prompts. Send them to your reader, then re-run with --score FILE.",
              file=sys.stderr)
        return 0
    mine = verdicts(a.score, len(rows))
    n = len(mine)
    print(f"your reader: {sum(mine)}/{n} = {sum(mine) / n:.4f} follow the passage")
    for key, label in (("qwen", "Qwen3-32B under data/rubrics/multihop.txt"),
                       ("reading", "our reading, from the kappa sample")):
        other = [r[key] for r in rows]
        if any(o is None for o in other):
            continue
        po, ka = cohen_kappa(mine, other)
        dis = sum(1 for x, y in zip(mine, other) if x != y)
        print(f"  vs {label}: {sum(other)}/{n} = {sum(other) / n:.4f}, agreement {po:.4f}, "
              f"kappa {ka:.4f}, {dis} disagreements")
    return 0


def cmd_judge(a) -> int:
    from contextgcg.replay import relabel
    return relabel(DATA, a.condition, limit=a.limit)


def cmd_answers(a) -> int:
    from contextgcg.replay import regenerate
    return regenerate(DATA, a.condition, limit=a.limit, out=Path(a.out) if a.out else None,
                      dry_run=a.dry_run)


def cmd_search(a) -> int:
    from contextgcg.replay import search
    return search(DATA, a.experiment, a.model, out=Path(a.out), limit=a.limit, seed=a.seed,
                  items=a.items, nonprogressive=a.nonprogressive)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("conditions"); p.set_defaults(f=cmd_conditions)
    p = sub.add_parser("compare")
    p.add_argument("condition_a"); p.add_argument("condition_b")
    p.add_argument("--bootstrap", action="store_true",
                   help="also resample passages, as the multi-hop table does")
    p.set_defaults(f=cmd_compare)
    p = sub.add_parser("read")
    p.add_argument("condition", help='section/model/condition, or "sample" for the rows carrying both labels')
    p.add_argument("--score", type=Path, metavar="FILE",
                   help="read A/B verdicts back and report agreement, instead of writing prompts")
    p.add_argument("--limit", type=int, default=0)
    p.set_defaults(f=cmd_read)
    p = sub.add_parser("judge"); p.add_argument("condition"); p.add_argument("--limit", type=int, default=0)
    p.set_defaults(f=cmd_judge)
    p = sub.add_parser("answers"); p.add_argument("condition"); p.add_argument("--limit", type=int, default=0)
    p.add_argument("--out", help="also write the regenerated rows here")
    p.add_argument("--dry-run", action="store_true", help="build the prompts with the tokenizer only")
    p.set_defaults(f=cmd_answers)
    p = sub.add_parser("search")
    p.add_argument("experiment",
                   choices=["per_item", "universal", "multihop", "single", "joint", "control"])
    p.add_argument("model", choices=MODELS)
    p.add_argument("--out", default="runs"); p.add_argument("--limit", type=int, default=0)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--items", choices=["basefakepedia", "cubnq"], default=None, help="per_item only")
    p.add_argument("--nonprogressive", action="store_true", help="universal only")
    p.set_defaults(f=cmd_search)
    a = ap.parse_args()
    try:
        return a.f(a)
    except (ReplayError, ReadingError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
