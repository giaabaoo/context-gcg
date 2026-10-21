from __future__ import annotations

import fnmatch
import json
from dataclasses import dataclass
from functools import cached_property, lru_cache
from pathlib import Path

import yaml

from contextgcg import models, prompting as P
from contextgcg.data import Item, load_items
from contextgcg.io import read_jsonl, write_jsonl
from contextgcg.stats import cohen_kappa

CONFIGS = Path(__file__).resolve().parent.parent / "configs"


class ReplayError(Exception):
    pass


@lru_cache(maxsize=None)
def _conditions() -> list[dict]:
    return [rule for f in sorted((CONFIGS / "conditions").glob("*.yml"))
            for rule in yaml.safe_load(f.read_text())]


@lru_cache(maxsize=None)
def _search() -> dict:
    out: dict = {}
    for f in sorted((CONFIGS / "search").glob("*.yml")):
        for k, v in yaml.safe_load(f.read_text()).items():
            if k in out:
                raise ReplayError(f"experiment {k!r} defined in two files under configs/search/")
            out[k] = v
    return out


def recipe(condition: str) -> dict:
    for rule in _conditions():
        if fnmatch.fnmatch(condition, rule["match"]):
            return rule
    raise ReplayError(f"no recipe for {condition} under configs/conditions/")


@dataclass
class Condition:
    section: str
    model: str
    name: str
    rule: dict
    rows: list[dict]
    release: Path

    @classmethod
    def load(cls, release: Path, condition: str, limit: int = 0) -> "Condition":
        section, model, name = condition.split("/")
        rule = recipe(condition)
        rows = read_jsonl(release / "runs" / section / model / f"{name}.jsonl")
        if limit:
            keep = set(list(dict.fromkeys(r["item_id"] for r in rows))[:limit])
            rows = [r for r in rows if r["item_id"] in keep]
        return cls(section, model, name, rule, rows, release)

    @cached_property
    def items(self) -> dict[str, Item]:
        named = self.rule.get("items")
        if not named:
            raise ReplayError(f"{self}: its recipe names no item pool "
                              f"({self.rule.get('why', 'not replayable')})")
        names = named if isinstance(named, list) else [named]
        return {it.id: it for n in names
                for it in load_items(self.release / "items" / n / f"{self.model}.jsonl")}

    def __str__(self) -> str:
        return f"{self.section}/{self.model}/{self.name}"

    @property
    def spec(self) -> models.ModelSpec:
        return models.MODELS[self.model]

    @property
    def layout(self) -> str:
        return self.rule["layout"]

    def item(self, row: dict) -> Item:
        return self.items[row["item_id"]]


def _slot(condition: Condition, row: dict) -> str:
    kind = condition.rule.get("slot", "none")
    if condition.rule.get("slot_from_name"):
        kind = "none" if condition.name.endswith("nosuffix") else "suffix"
    return {"suffix": row.get("suffix") or "",
            "instruction": row.get("instruction") or "",
            "none": ""}[kind]


def user_message(condition: Condition, row: dict) -> str:
    slot = _slot(condition, row)
    passage = condition.item(row).meta[condition.rule["passage"]] if condition.rule.get("passage") \
        else condition.item(row).passage
    query = row.get("question") or row.get("query") or condition.item(row).query
    if condition.layout in ("doc", "reask"):
        return P.doc(passage, query, slot)
    if condition.layout == "doc_tight":
        return P.doc_tight(passage, query, slot)
    if condition.layout == "instruction_above":
        return P.instruction_above(slot or None, passage, query, condition.rule.get("gap", "\n"))
    if condition.layout == "passage_deleted":
        return P.passage_deleted(slot, query)
    raise ReplayError(f"layout {condition.layout} has no single user turn")


def messages(condition: Condition, row: dict) -> list[dict]:
    if condition.layout == "minder":
        return P.minder(condition.item(row).passage, row.get("question") or condition.item(row).query,
                        condition.rule.get("format_line", False))
    return P.user(user_message(condition, row))


def _reask_conversations(condition: Condition) -> tuple[list[list[dict]], list[dict]]:
    by_item: dict = {}
    for row in condition.rows:
        by_item.setdefault(row["item_id"], {})[row["turn"]] = row
    convs, rows = [], []
    for turns in by_item.values():
        first = user_message(condition, turns[1])
        query = condition.item(turns[1]).query
        if condition.rule.get("regen_turn1"):
            convs.append(P.user(first))
            rows.append(turns[1])
        second = P.reask(first, turns[1]["answer"], query)
        convs.append(second)
        rows.append(turns[2])
        convs.append(second + [{"role": "assistant", "content": turns[2]["answer"]},
                               {"role": "user", "content": query}])
        rows.append(turns[3])
    return convs, rows


def conversations(condition: Condition) -> tuple[list[list[dict]], list[dict]]:
    if condition.layout == "reask":
        return _reask_conversations(condition)
    return [messages(condition, row) for row in condition.rows], condition.rows


def relabel(release: Path, condition_name: str, limit: int = 0) -> int:
    from contextgcg.judge import JUDGE_REPO, QwenJudge
    condition = Condition.load(release, condition_name, limit)
    if not condition.rule.get("judge"):
        raise ReplayError(f"{condition}: scored by reading only, no rubric applies")
    net, tok = models.load(models.ModelSpec("qwen", JUDGE_REPO, "", False, None))
    judge = QwenJudge(net, tok, condition.rule["judge"])
    fresh, shipped = [], []
    for row in condition.rows:
        it = condition.item(row)
        fresh.append(judge(passage=it.passage,
                           query=row.get("question") or row.get("query") or it.query,
                           planted=row.get("target") or row.get("ck") or it.planted,
                           true=row.get("pk") or it.true, answer=row["answer"],
                           claim=it.meta.get("claim", "")))
        shipped.append(row["label"])
    paired = [(f, s) for f, s in zip(fresh, shipped) if isinstance(s, bool)]
    rate = f"{condition}: Qwen {sum(fresh)}/{len(fresh)} = {sum(fresh) / len(fresh):.3f}"
    if not paired:
        print(f"{rate}; nothing to compare it with, this condition carries no stored label")
        return 0
    agreement, kappa = cohen_kappa([f for f, _ in paired], [s for _, s in paired])
    print(f"{rate}; shipped {sum(s for _, s in paired)}/{len(paired)}; "
          f"agreement {agreement:.4f}, kappa {kappa:.4f}")
    return 0


def _steering_hooks(condition: Condition, net):
    if not condition.rule.get("steer"):
        return None
    from contextgcg.baselines.steering import clamp, load_direction
    knob = condition.spec.knob
    direction = load_direction(knob.repo, CONFIGS / "knob" / f"{condition.model}.pt")
    return lambda: clamp(net, direction, knob.layer, knob.context_value)


def _cad_answers(condition: Condition, net, tok) -> list[tuple[dict, str]]:
    from contextgcg.baselines.cad import CAD
    cad = CAD(net, tok, condition.rule.get("max_new", 256))
    return [(row, cad(condition.item(row), condition.rule["alpha"], condition.rule.get("chat", True)))
            for row in condition.rows]


def _generated_answers(condition: Condition, net, tok) -> list[tuple[dict, str]]:
    from contextgcg.generate import Generator
    gen = Generator(net, tok, condition.rule.get("max_new", 256), condition.rule.get("batch", 8),
                    condition.spec.eot if condition.rule.get("stop_at_eot") else None)
    convs, rows = conversations(condition)
    return list(zip(rows, gen(convs, hooks=_steering_hooks(condition, net))))


def regenerate(release: Path, condition_name: str, limit: int = 0, out: Path | None = None,
               dry_run: bool = False) -> int:
    condition = Condition.load(release, condition_name, limit)
    if condition.rule.get("regen") is False:
        raise ReplayError(f"{condition}: {condition.rule.get('why', 'not regenerable from the shipped answers')}")
    if dry_run:
        return print_prompts(condition)
    net, tok = models.load(condition.spec)
    done = _cad_answers(condition, net, tok) if condition.layout == "cad" else _generated_answers(condition, net, tok)
    same = sum(answer == row["answer"] for row, answer in done)
    note = "" if condition.rule.get("verified", True) else f"  (settings inferred: {condition.rule.get('why', '')})"
    print(f"{condition}: {same}/{len(done)} answers identical to the shipped ones{note}")
    if out:
        write_jsonl(out, [{**row, "regenerated": answer} for row, answer in done])
    return 0


def print_prompts(condition: Condition) -> int:
    tok = models.load_tokenizer(condition.spec)
    if condition.layout == "cad":
        from contextgcg.baselines.cad import prompts
        texts = [prompts(tok, condition.item(row), condition.rule.get("chat", True))[0] for row in condition.rows]
    else:
        convs, _ = conversations(condition)
        texts = [P.render(tok, c) for c in convs]
    print(f"{condition.model}/{condition.name}: {len(texts)} prompts built; first:\n{texts[0]}")
    return 0


def _bank_entry(item: dict) -> dict:
    questions = item["meta"]["questions"]
    return {"id": item["id"], "passage": item["passage"], "base_question": item["query"],
            "planted": item["planted"],
            "train": [q for q in questions if q["split"] == "train"],
            "test": [q for q in questions if q["split"] != "train"]}


def _append(path: Path, row: dict) -> None:
    with path.open("a") as f:
        f.write(json.dumps(row) + "\n")


def search(release: Path, experiment: str, model: str, out: Path, limit: int = 0,
           seed: int | None = None, items: str | None = None, nonprogressive: bool = False) -> int:
    cfg = _search()[experiment]
    out = out / experiment / model
    out.mkdir(parents=True, exist_ok=True)
    spec = models.MODELS[model]

    def pool(dataset: str) -> list[Item]:
        got = load_items(release / "items" / dataset / f"{model}.jsonl")
        return got[:limit] if limit else got

    if experiment == "per_item":
        from contextgcg.experiments import per_item
        dataset = items or cfg["items"]
        s = seed if seed is not None else (0 if dataset == "cubnq" else cfg["seeds"][model])
        per_item.search(model, pool(dataset), out / f"{dataset}_seed{s}.jsonl", seed=s,
                        steps=cfg["steps"])

    elif experiment == "universal":
        from contextgcg.experiments import universal
        from contextgcg.optim.universal import UniversalConfig
        split = json.loads((release / "items" / "universal_train" / "splits.json").read_text())[model]
        source, ids = (("basefakepedia", split["train20"]) if nonprogressive
                       else ("universal_train", split["progressive_train_ids"]))
        by_id = {it.id: it for it in load_items(release / "items" / source / f"{model}.jsonl")}
        cfg_u = UniversalConfig(target=cfg["models"][model]["target"], steps=cfg["steps"],
                                progressive=not nonprogressive)
        universal.search(model, [by_id[i] for i in ids], cfg_u,
                         out / ("nonprogressive" if nonprogressive else "progressive"))

    elif experiment in ("multihop", "single"):
        from contextgcg.experiments import multihop
        bank = [_bank_entry(r) for r in
                read_jsonl(release / "items" / "entailed" / f"{model}.jsonl")]
        net, tok = models.load(spec)
        for entry in (bank[:limit] if limit else bank):
            _append(out / "rows.jsonl",
                    multihop.run(model, entry, single=experiment == "single", model=net, tok=tok))

    elif experiment in ("joint", "control"):
        from contextgcg.experiments import multiturn
        by_id = {it.id: it for it in load_items(release / "items" / "repair" / f"{model}.jsonl")}
        shipped = read_jsonl(release / "runs" / "repair" / model / f"{experiment}.jsonl")
        ids = list(dict.fromkeys(r["item_id"] for r in shipped))
        net, tok = models.load(spec)
        for item_id in (ids[:limit] if limit else ids):
            _append(out / "rows.jsonl",
                    multiturn.repair(model, by_id[item_id], cfg["targets"][model], experiment,
                                     seed=1, model=net, tok=tok))

    print("wrote", out)
    return 0
