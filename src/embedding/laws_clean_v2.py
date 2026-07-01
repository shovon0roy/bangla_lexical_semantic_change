"""Stage 6b: de-confound the two laws — frequency-stratified + name-excluded.

The raw Law-2 result (negative rho) is suspected to be a PROPER-NOUN artifact: names
have tight neighbourhoods (low dispersion) AND high ChangeScore, manufacturing a
spurious 'low dispersion -> high change' trend. This script tests that directly by
recomputing both laws while (a) excluding flagged names and (b) raising the frequency
floor. If Law 2's negative rho attenuates toward 0 as the vocab is cleaned, it was an
artifact; a robust law would persist.

Reuses results_v2/laws_data.csv (already has dispersion) + results_v2/drift_*.csv
(count_a/count_b) + the seed-name propagation on the early model. Deterministic.

Output: results_v2/laws_clean_summary.json  (+ console table)

Run:  .venv/bin/python src/embedding/laws_clean_v2.py
"""
import os
import sys
if os.environ.get("PYTHONHASHSEED") != "0":
    os.environ["PYTHONHASHSEED"] = "0"
    os.execv(sys.executable, [sys.executable] + sys.argv)

import contextlib  # noqa: E402
import csv  # noqa: E402
import json  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402
from gensim.models import Word2Vec  # noqa: E402
from bangla_stemmer.stemmer import stemmer  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results_v2"
MODELS = ROOT / "models_v2"
SEED_FILE = ROOT / "corpus_build" / "bn_name_seed.txt"
DRIFT = RESULTS / "drift_1950_1970_2010_2025.csv"
LAWS_DATA = RESULTS / "laws_data.csv"
EARLY = "1950_1970"

_dn = open(os.devnull, "w")
_S = stemmer.BanglaStemmer()


def stem(w):
    with contextlib.redirect_stdout(_dn):
        try:
            return _S.stem(w) or w
        except Exception:
            return w


def name_set(kv, thresh=0.3, rounds=8, knn=20):
    seed = set()
    for ln in SEED_FILE.read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if ln and not ln.startswith("#"):
            seed.add(stem(ln))
    names = {w for w in seed if w in kv}
    cand = [w for w in kv.index_to_key if kv.get_vecattr(w, "count") >= 10]
    nbr = {w: [x for x, _ in kv.most_similar(w, topn=knn)] for w in cand}
    for _ in range(rounds):
        add = {w for w in cand if w not in names and nbr[w]
               and sum(n in names for n in nbr[w]) / len(nbr[w]) >= thresh}
        if not add:
            break
        names |= add
    return names


def law(records, key):
    x = np.array([r[key] for r in records]); y = np.array([r["change"] for r in records])
    rho, p = spearmanr(x, y)
    return round(float(rho), 4), float(p), len(records)


def main():
    # dispersion + logf + change (already computed)
    data = {r["word"]: r for r in csv.DictReader(open(LAWS_DATA, encoding="utf-8-sig"))}
    # counts from drift
    cnt = {r["word"]: (int(r["count_a"]), int(r["count_b"]))
           for r in csv.DictReader(open(DRIFT, encoding="utf-8-sig"))}
    print("computing name set on early model (propagation) ...", file=sys.stderr)
    kv = Word2Vec.load(str(MODELS / f"word2vec_{EARLY}.model")).wv
    names = name_set(kv)
    print(f"flagged names: {len(names)}", file=sys.stderr)

    recs = []
    for w, d in data.items():
        if w not in cnt:
            continue
        ca, cb = cnt[w]
        recs.append({"word": w, "logf": float(d["log_frequency"]),
                     "disp": float(d["polysemy_dispersion"]), "change": float(d["change_score"]),
                     "ca": ca, "cb": cb, "name": w in names})

    print(f"\n{'config':28} {'n':>6}  {'Law1 freq rho (p)':>26}  {'Law2 polysemy rho (p)':>26}")
    summary = {"flagged_names": len(names), "configs": {}}
    configs = [("all words (raw)", 0, False),
               ("names excluded", 0, True),
               ("freq>=50, names excl", 50, True),
               ("freq>=100, names excl", 100, True),
               ("freq>=200, names excl", 200, True),
               ("freq>=500, names excl", 500, True)]
    for label, floor, excl in configs:
        sub = [r for r in recs if r["ca"] >= floor and r["cb"] >= floor and not (excl and r["name"])]
        if len(sub) < 30:
            continue
        r1, p1, n = law(sub, "logf")
        r2, p2, _ = law(sub, "disp")
        s1 = f"{r1:+.4f} ({p1:.1e})"; s2 = f"{r2:+.4f} ({p2:.1e})"
        print(f"{label:28} {n:>6}  {s1:>26}  {s2:>26}")
        summary["configs"][label] = {"n": n, "floor": floor, "names_excluded": excl,
                                     "law1_rho": r1, "law1_p": p1, "law2_rho": r2, "law2_p": p2}
    (RESULTS / "laws_clean_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                                                     encoding="utf-8")
    print(f"\nwrote laws_clean_summary.json -> {RESULTS}")
    print("\nInterpretation: if Law-2 rho moves toward 0 as the vocab is cleaned, the raw")
    print("negative result was a proper-noun / low-frequency artifact (paper's NULL holds).")


if __name__ == "__main__":
    main()
