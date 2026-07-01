"""Flag proper nouns (mostly person names) in the semantic-change ranking and emit
filtered top-changed lists — names dominate the raw ranking in a literary corpus
because a character name lands in different novels each era (its neighbourhood flips),
which is NOT meaning change.

Method (deterministic, unsupervised beyond a small seed):
  * load a high-precision seed of common Bengali names (corpus_build/bn_name_seed.txt),
    stemmed to match the corpus
  * PROPAGATE in embedding space: names cluster tightly, so any word whose top-K
    neighbours are >= THRESH already-flagged names is itself flagged. Iterate a few
    rounds over each era model; take the union across eras.
  * a word in the change ranking is dropped if it is in the flagged name set.

For each pair, writes results_v2/top_changed_clean_<A>_<B>.csv (names removed) and
records how many were filtered. The raw drift_*.csv is left untouched (transparency).

Run:  .venv/bin/python src/embedding/filter_propernouns_v2.py   # --thresh 0.4 --rounds 3 --knn 15
"""
import os
import sys
if os.environ.get("PYTHONHASHSEED") != "0":
    os.environ["PYTHONHASHSEED"] = "0"
    os.execv(sys.executable, [sys.executable] + sys.argv)

import argparse  # noqa: E402
import contextlib  # noqa: E402
import csv  # noqa: E402
import json  # noqa: E402
from pathlib import Path  # noqa: E402

from gensim.models import Word2Vec  # noqa: E402
from bangla_stemmer.stemmer import stemmer  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
MODELS = ROOT / "models_v2"
RESULTS = ROOT / "results_v2"
SEED_FILE = ROOT / "corpus_build" / "bn_name_seed.txt"
ERAS = ["pre_1950", "1950_1970", "1970_1990", "1990_2010", "2010_2025"]
PAIRS = [(ERAS[i], ERAS[i + 1]) for i in range(len(ERAS) - 1)] + [("1950_1970", "2010_2025")]

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
    """Grow the name set inside one model: a word counts as a name if >= thresh of its
    knn neighbours are already names. Neighbours are computed ONCE and cached, so many
    propagation rounds are cheap -> aggressive cascade through name clusters."""
    names = {w for w in names if w in kv}
    cand = [w for w in kv.index_to_key if kv.get_vecattr(w, "count") >= 10]
    nbr = {w: [x for x, _ in kv.most_similar(w, topn=knn)] for w in cand}  # cache once
    for _ in range(rounds):
        added = {w for w in cand
                 if w not in names and nbr[w]
                 and sum(n in names for n in nbr[w]) / len(nbr[w]) >= thresh}
        if not added:
            break
        names |= added
    return names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--thresh", type=float, default=0.4)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--knn", type=int, default=15)
    ap.add_argument("--topn", type=int, default=50)
    args = ap.parse_args()

    seed = load_seed()
    print(f"seed names (stemmed): {len(seed)}")

    # build the flagged-name set per era (cached), union as needed per pair
    name_by_era = {}
    for era in ERAS:
        kv = Word2Vec.load(str(MODELS / f"word2vec_{era}.model")).wv
        names = propagate(kv, seed, args.thresh, args.rounds, args.knn)
        name_by_era[era] = names
        print(f"  {era:10} flagged names: {len(names):,}  (seed in vocab + propagated)")

    summary = {"params": vars(args), "seed": len(seed), "pairs": {}}
    for a, b in PAIRS:
        names = name_by_era[a] | name_by_era[b]
        src = RESULTS / f"drift_{a}_{b}.csv"
        if not src.exists():
            continue
        rows = list(csv.DictReader(open(src, encoding="utf-8-sig")))
        clean = [r for r in rows if r["word"] not in names]
        removed = len(rows) - len(clean)
        # already sorted by change_score desc in drift_*.csv
        with open(RESULTS / f"top_changed_clean_{a}_{b}.csv", "w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(clean[:50])
        summary["pairs"][f"{a}_{b}"] = {"total": len(rows), "names_removed": removed,
                                        "kept": len(clean)}
        print(f"\n=== {a} -> {b} ===  removed {removed} name-words of {len(rows)}")
        print(f"  TOP changed (clean): {', '.join(r['word'] for r in clean[:15])}")

    (RESULTS / "namefilter_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                                                     encoding="utf-8")
    print(f"\nwrote top_changed_clean_*.csv + namefilter_summary.json -> {RESULTS}")


if __name__ == "__main__":
    main()
