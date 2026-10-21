"""Put a suffix on your own passage: ours, or one searched for your passage.

    cp data/example_passage.yml mine.yml      then edit the three fields
    python run_suffix.py --model llama --input mine.yml --universal
    python run_suffix.py --model llama --input mine.yml

    python run_suffix.py --model llama --example
    python run_suffix.py --model llama --example --dry-run    prints the prompts, no GPU

Flags work too, for a quick try:

    python run_suffix.py --model llama --passage "..." --question "..." --planted Lisbon

--universal appends one of the suffixes we fitted for that model and skips the search. Without it,
GCG optimizes a 20-token suffix for your passage, about a minute on one B200. Either way the
question is asked twice, with the suffix and without, and both answers are printed.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import yaml

from contextgcg import models
from contextgcg.data import Item, load_items
from contextgcg.generate import Generator
from contextgcg.optim.gcg import SearchConfig, SuffixSearch
from contextgcg.prompting import doc, render, user

DATA = Path(__file__).resolve().parent / "data"


def read_input(path: Path) -> Item:
    try:
        d = yaml.safe_load(path.read_text())
    except yaml.YAMLError as e:
        raise SystemExit(f"{path}: not valid YAML.\n{e}\n\n"
                         f"The passage needs a block scalar: 'passage: |' with the text indented "
                         f"under it. See data/example_passage.yml.")
    if isinstance(d, dict):
        d = {("true" if k is True else str(k)): v for k, v in d.items()}
    if not isinstance(d, dict):
        raise SystemExit(f"{path}: expected three fields, got {type(d).__name__}. "
                         f"Copy data/example_passage.yml and edit it.")
    missing = [k for k in ("passage", "question", "planted") if not str(d.get(k) or "").strip()]
    if missing:
        raise SystemExit(f"{path}: missing or empty: {', '.join(missing)}")
    extra = sorted(set(d) - {"passage", "question", "planted", "true"})
    if extra:
        raise SystemExit(f"{path}: unknown field(s): {', '.join(extra)}")
    return Item("yours", str(d["question"]).strip(), str(d["passage"]).strip(),
                str(d["planted"]).strip(), str(d.get("true") or "").strip())


BEST_UNIVERSAL = {"llama": "progressive", "mistral": "nonprog", "gemma": "nonprog"}


def universal_suffix(model: str) -> tuple[str, str]:
    condition = BEST_UNIVERSAL[model]
    path = DATA / "items" / "universal_train" / "suffixes.json"
    return json.loads(path.read_text())[model][condition]["suffix"], condition


def check_input(it: Item) -> None:
    if it.planted.lower() not in it.passage.lower():
        print(f"warning: {it.planted!r} does not appear in the passage.")
    if it.query.rstrip().endswith("?"):
        print("warning: the question ends in '?'. The layout expects a stem the answer completes, "
              "such as 'The Eiffel Tower is located in'.")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, choices=list(models.MODELS))
    p.add_argument("--passage", help="the passage the model will read")
    p.add_argument("--question", help="the question, phrased so the answer completes it")
    p.add_argument("--planted", help="the answer the passage states")
    p.add_argument("--input", type=Path, metavar="FILE",
                   help="a YAML file with passage, question and planted")
    p.add_argument("--example", action="store_true", help="use the first item of our own data")
    p.add_argument("--universal", action="store_true",
                   help="use our fitted universal suffix instead of searching for yours")
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--dry-run", action="store_true", help="print the prompts; no model, no GPU")
    a = p.parse_args()

    if a.input:
        if not a.input.exists():
            p.error(f"{a.input} not found. Start from data/example_passage.yml")
        it = read_input(a.input)
        check_input(it)
    elif a.example:
        it = load_items(DATA / "items" / "basefakepedia" / f"{a.model}.jsonl")[0]
    elif a.passage and a.question and a.planted:
        it = Item("yours", a.question, a.passage, a.planted, "")
        check_input(it)
    else:
        p.error("give --input FILE, or --passage with --question and --planted, or --example")

    spec = models.MODELS[a.model]
    tok = models.load_tokenizer(spec)
    cfg = SearchConfig(steps=a.steps)
    search = SuffixSearch(None, tok, spec.eot, spec.prefix_cache, cfg)
    universal, universal_schedule = universal_suffix(a.model) if a.universal else (None, "")

    print(f"passage : {it.passage}")
    print(f"question: {it.query}")
    print(f"planted : {it.planted}")
    if universal:
        print(f"\nour universal suffix for {a.model} ({universal_schedule}), appended to your passage:")
        print("-" * 80)
        print(render(tok, user(doc(it.passage, it.query, universal))))
    else:
        print(f"\nThe search optimizes the marked slot so the answer becomes {search.target(it)!r}:")
        print("-" * 80)
        print(render(tok, user(search.message(it))))
    print("-" * 80)

    if a.dry_run:
        stop = "to use it" if universal else "to run the search"
        print(f"\n[dry-run] stopping here. Drop --dry-run {stop}.")
        return

    model, tok = models.load(spec)
    search.model = model

    gen = Generator(model, tok, max_new_tokens=256, batch_size=2)
    floor, no_passage = gen([user(doc(it.passage, it.query, "")), user(it.query)])
    print(f"\nwith the passage, no suffix: {floor!r}")
    print(f"the question alone         : {no_passage!r}")
    print(f"\nIf that first answer already gives {it.planted!r}, a suffix has nothing to change.")

    if universal:
        suffix = universal
        print(f"\nusing our universal suffix ({universal_schedule}), no search: {suffix!r}")
    else:
        print(f"\nsearching, {a.steps} steps ...")
        res = search(it)
        suffix = res.suffix
        print(f"suffix: {suffix!r}\nfinal loss: {res.loss:.4f}")

    with_suffix = gen([user(doc(it.passage, it.query, suffix))])[0]
    print(f"\nwithout the suffix: {floor!r}")
    print(f"with the suffix   : {with_suffix!r}")
    if it.true:
        print(f"\n(our data records the model's own answer as {it.true!r})")
    print("\nThis counts as a success if the answer gives the planted fact and does not deny it.")


if __name__ == "__main__":
    main()
