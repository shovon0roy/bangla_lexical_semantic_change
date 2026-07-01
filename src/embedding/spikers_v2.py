"""v2 Spikers: words nearly absent in the early eras that surge in the modern era,
ranked by modern smoothed frequency, with proper-noun (person/character name) leakage
removed via the same seed+propagation flagger used in filter_propernouns_v2.py.

A literary corpus makes raw frequency spikers dominated by character/person names
(each new novel introduces new names); those are not lexical change, so we drop them.

Writes results_v2/spikers_clean.csv (rank, word, modern_pm, early_pm) and prints the top 12.

Run:  PYTHONHASHSEED=0 .venv/bin/python src/embedding/spikers_v2.py
"""
import os
import sys
if os.environ.get("PYTHONHASHSEED") != "0":
    os.environ["PYTHONHASHSEED"] = "0"
    os.execv(sys.executable, [sys.executable] + sys.argv)

import contextlib  # noqa: E402
import csv  # noqa: E402
from collections import Counter  # noqa: E402
from pathlib import Path  # noqa: E402

from gensim.models import Word2Vec  # noqa: E402
from bangla_stemmer.stemmer import stemmer  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
MODELS = ROOT / "models_v2"
DATA = ROOT / "data_cleaned_v2"
RESULTS = ROOT / "results_v2"
SEED_FILE = ROOT / "corpus_build" / "bn_name_seed.txt"
ERAS = ["pre_1950", "1950_1970", "1970_1990", "1990_2010", "2010_2025"]
EARLY_ERAS = ["pre_1950", "1950_1970", "1970_1990"]   # "early" = pre-1990
THRESH, ROUNDS, KNN = 0.3, 8, 20                       # same as the pipeline name-filter run
EARLY_MAX, MIN_MODERN = 1.0, 5.0                       # nearly absent early; present now (per million)

_devnull = open(os.devnull, "w")
_S = stemmer.BanglaStemmer()


def stem(w):
    with contextlib.redirect_stdout(_devnull):
        try:
            return _S.stem(w) or w
        except Exception:
            return w


def load_seed():
    out = set()
    for line in SEED_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            out.add(stem(line))
    return out


def propagate(kv, names, thresh, rounds, knn):
    names = {w for w in names if w in kv}
    cand = [w for w in kv.index_to_key if kv.get_vecattr(w, "count") >= 10]
    nbr = {w: [x for x, _ in kv.most_similar(w, topn=knn)] for w in cand}
    for _ in range(rounds):
        added = {w for w in cand
                 if w not in names and nbr[w]
                 and sum(n in names for n in nbr[w]) / len(nbr[w]) >= thresh}
        if not added:
            break
        names |= added
    return names


def per_million():
    pm = {}
    for era in ERAS:
        toks = [t for t in (DATA / f"{era}.txt").read_text(encoding="utf-8").split()
                if t not in ("<s>", "<s/>")]
        n = len(toks); c = Counter(toks)
        pm[era] = {w: cnt / n * 1e6 for w, cnt in c.items()}
    return pm


def main():
    seed = load_seed()
    print(f"seed names (stemmed): {len(seed)}")
    names = set()
    for era in ERAS:
        kv = Word2Vec.load(str(MODELS / f"word2vec_{era}.model")).wv
        flagged = propagate(kv, seed, THRESH, ROUNDS, KNN)
        names |= flagged
        print(f"  {era:10} flagged names: {len(flagged):,}")
    print(f"union flagged names: {len(names):,}")

    pm = per_million()
    early = lambda w: max(pm[e].get(w, 0.0) for e in EARLY_ERAS)
    modern = pm["2010_2025"]
    cand = [(w, modern[w], early(w)) for w in modern
            if early(w) < EARLY_MAX and modern[w] >= MIN_MODERN
            and len(w) > 1 and w not in names]
    cand.sort(key=lambda x: -x[1])

    out = RESULTS / "spikers_clean.csv"
    with open(out, "w", encoding="utf-8-sig", newline="") as f:
        wr = csv.writer(f); wr.writerow(["rank", "word", "modern_pm", "early_pm"])
        for i, (w, m, e) in enumerate(cand[:40], 1):
            wr.writerow([i, w, f"{m:.2f}", f"{e:.3f}"])
    print(f"\nwrote {out}  ({len(cand)} spikers after name filter)")
    print("rank  word            modern_pm  early_pm")
    for i, (w, m, e) in enumerate(cand[:20], 1):
        print(f"{i:>3}  {w:<14}  {m:>8.1f}  {e:>6.3f}")


if __name__ == "__main__":
    main()
