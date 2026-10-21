from __future__ import annotations

from math import comb


def mcnemar_exact(won: int, lost: int) -> float:
    n = won + lost
    if n == 0:
        return 1.0
    k = min(won, lost)
    tail = sum(comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def paired_discordant(a: dict, b: dict, positive: object = True) -> tuple[int, int, int]:
    common = sorted(set(a) & set(b))
    won = sum(1 for k in common if a[k] == positive and b[k] != positive)
    lost = sum(1 for k in common if b[k] == positive and a[k] != positive)
    return won, lost, len(common)


def cohen_kappa(x: list, y: list) -> tuple[float, float]:
    assert len(x) == len(y) and x
    n = len(x)
    po = sum(1 for a, b in zip(x, y) if a == b) / n
    cats = set(x) | set(y)
    pe = sum((x.count(c) / n) * (y.count(c) / n) for c in cats)
    return po, (po - pe) / (1 - pe) if pe < 1 else 1.0


def passage_bootstrap(a: dict, b: dict, reps: int = 6000, seed: int = 0) -> dict:
    import random
    ks = sorted(set(a) & set(b))
    byp: dict = {}
    for x in ks:
        byp.setdefault(x[0], []).append((a[x], b[x]))
    pk = list(byp)
    rng = random.Random(seed)
    d = []
    for _ in range(reps):
        s = [byp[rng.choice(pk)] for _ in pk]
        q = [y for g in s for y in g]
        d.append(sum(u for u, _ in q) / len(q) - sum(v for _, v in q) / len(q))
    d.sort()
    lo_i, hi_i = int(0.025 * reps), int(0.975 * reps)
    return {"diff": sum(d) / len(d), "lo": d[lo_i], "hi": d[hi_i],
            "p": 2 * min(sum(x <= 0 for x in d), sum(x >= 0 for x in d)) / len(d),
            "reps": len(d),
            "n": len(ks), "passages": len(pk)}
