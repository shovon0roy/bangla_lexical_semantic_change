"""An artifact-free test of the Law of Polysemy (Law of Innovation) on Bangla.

The paper's earlier tests derive BOTH polysemy and change from the same static
embedding space, which Dubossarsky et al. (2017) showed can manufacture or mask the
laws. This script replaces the fragile side(s) with independent measurements:

  polysemy  = number of IndoWordNet (Bengali) synsets containing the word
              (lexicographic sense count; no embedding geometry involved)
  change A  = the pipeline's ChangeScore (embedding-based, for comparability)
  change B  = Jensen-Shannon divergence between the word's collocate distributions
              in 1950-1970 vs 2010-2025 (pure counts; no embeddings anywhere)

Test A  (senses vs ChangeScore) removes the genericness confound of the dispersion
proxy. Test B (senses vs collocate-JSD) removes embeddings from both variables, so
nothing the models put into the geometry can produce the correlation. Frequency is
controlled by rank-partial correlation and by frequency-tertile breakdown, since
sense counts and distributional noise both track frequency.

Deterministic (no sampling). Requires: pip install pyiwn (downloads IndoWordNet on
first use).

Output: results_v2/law2_wordnet_summary.json (+ console table)
Run:    .venv/bin/python src/embedding/law2_wordnet_v2.py
"""
import csv
import json
import math
import sys
import unicodedata
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr, rankdata

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results_v2"
DATA = ROOT / "data_cleaned_v2"
ERA_A, ERA_B = "1950_1970", "2010_2025"
WINDOW = 5           # same context window as the collocate module
MIN_OCC = 20         # a word needs >=20 occurrences in BOTH eras for a stable JSD

nfc = lambda s: unicodedata.normalize("NFC", s)


def wordnet_senses():
    import pyiwn
    iwn = pyiwn.IndoWordNet(lang=pyiwn.Language.BENGALI)
    sense = {}
    for w in iwn.all_words():
        if "_" in w:                       # single-word lemmas only
            continue
        try:
            sense[nfc(w)] = len(iwn.synsets(w))
        except Exception:
            pass
    return sense


def sentences(path):
    out, cur = [], []
    for tok in path.read_text(encoding="utf-8").split():
        if tok in ("<s>", "<s/>"):
            if cur:
                out.append(cur)
                cur = []
        else:
            cur.append(tok)
    if cur:
        out.append(cur)
    return out


def collocate_counts(era, targets):
    """One pass over an era: windowed collocate Counter for every target word."""
    co = {t: Counter() for t in targets}
    occ = Counter()
    for s in sentences(DATA / f"{era}.txt"):
        for i, t in enumerate(s):
            if t in co:
                occ[t] += 1
                lo, hi = max(0, i - WINDOW), min(len(s), i + WINDOW + 1)
                for c in s[lo:i] + s[i + 1:hi]:
                    if c != t:
                        co[t][c] += 1
    return co, occ


def jsd(p_counts, q_counts):
    """Jensen-Shannon divergence (base 2, in [0,1]) between two count distributions."""
    keys = set(p_counts) | set(q_counts)
    p = np.array([p_counts.get(k, 0) for k in keys], dtype=float)
    q = np.array([q_counts.get(k, 0) for k in keys], dtype=float)
    p /= p.sum()
    q /= q.sum()
    m = 0.5 * (p + q)
    def kl(a, b):
        mask = a > 0
        return float(np.sum(a[mask] * np.log2(a[mask] / b[mask])))
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def partial_spearman(x, y, z):
    rx, ry, rz = rankdata(x), rankdata(y), rankdata(z)
    def resid(a, b):
        b1 = np.vstack([b, np.ones_like(b)]).T
        coef, *_ = np.linalg.lstsq(b1, a, rcond=None)
        return a - b1 @ coef
    return float(np.corrcoef(resid(rx, rz), resid(ry, rz))[0, 1])


def report(name, x, y, logf, expect):
    rho, p = spearmanr(x, y)
    pr = partial_spearman(x, y, logf)
    # frequency tertiles
    terts = np.quantile(logf, [1 / 3, 2 / 3])
    strata = {}
    for lbl, mask in [("low_freq", logf <= terts[0]),
                      ("mid_freq", (logf > terts[0]) & (logf <= terts[1])),
                      ("high_freq", logf > terts[1])]:
        r, pp = spearmanr(np.asarray(x)[mask], np.asarray(y)[mask])
        strata[lbl] = {"rho": round(float(r), 4), "p": float(pp), "n": int(mask.sum())}
    print(f"\n=== {name} ===  (law predicts {expect})")
    print(f"  n = {len(x)}   Spearman rho = {rho:+.4f}   p = {p:.2e}   partial|freq = {pr:+.4f}")
    for lbl, d in strata.items():
        print(f"    {lbl:9}: rho = {d['rho']:+.4f}  p = {d['p']:.2e}  (n={d['n']})")
    return {"n": len(x), "spearman_rho": round(float(rho), 4), "p": float(p),
            "partial_rho_given_freq": round(pr, 4), "by_freq_tertile": strata}


def main():
    print("loading IndoWordNet (Bengali) sense counts ...")
    sense = wordnet_senses()
    print(f"  single-word lemmas: {len(sense):,}")

    rows = list(csv.DictReader(open(RESULTS / "laws_data.csv", encoding="utf-8-sig")))
    matched = [(nfc(r["word"]), sense[nfc(r["word"])], float(r["log_frequency"]),
                float(r["polysemy_dispersion"]), float(r["change_score"]))
               for r in rows if nfc(r["word"]) in sense]
    words = [m[0] for m in matched]
    senses = np.array([m[1] for m in matched], dtype=float)
    logf = np.array([m[2] for m in matched])
    disp = np.array([m[3] for m in matched])
    change = np.array([m[4] for m in matched])
    print(f"  matched against laws vocabulary: {len(words):,} "
          f"(monosemous {int((senses == 1).sum())}, polysemous {int((senses > 1).sum())})")

    out = {"wordnet_lemmas": len(sense), "matched": len(words),
           "eras": [ERA_A, ERA_B], "window": WINDOW, "min_occ_jsd": MIN_OCC, "tests": {}}

    # --- Test A: dictionary senses vs embedding ChangeScore ---------------------
    out["tests"]["A_senses_vs_changescore"] = report(
        "Test A: WordNet senses vs ChangeScore", senses, change, logf, "positive rho")

    # --- diagnostic: does the dispersion proxy track dictionary polysemy at all? -
    rho_d, p_d = spearmanr(disp, senses)
    print(f"\n=== Diagnostic: dispersion proxy vs WordNet senses ===")
    print(f"  Spearman rho = {rho_d:+.4f}   p = {p_d:.2e}   "
          f"(near zero = dispersion does not measure polysemy)")
    out["tests"]["diagnostic_dispersion_vs_senses"] = {
        "spearman_rho": round(float(rho_d), 4), "p": float(p_d)}

    # --- Test B: dictionary senses vs count-based collocate JSD -----------------
    print("\ncounting collocates for Test B (two corpus passes) ...")
    tset = set(words)
    co_a, occ_a = collocate_counts(ERA_A, tset)
    co_b, occ_b = collocate_counts(ERA_B, tset)
    keep, jsds = [], []
    for i, w in enumerate(words):
        if occ_a[w] >= MIN_OCC and occ_b[w] >= MIN_OCC:
            keep.append(i)
            jsds.append(jsd(co_a[w], co_b[w]))
    keep = np.array(keep)
    jsds = np.array(jsds)
    print(f"  words with >= {MIN_OCC} occurrences in both eras: {len(keep):,}")
    out["tests"]["B_senses_vs_collocate_jsd"] = report(
        "Test B: WordNet senses vs collocate JSD (no embeddings anywhere)",
        senses[keep], jsds, logf[keep], "positive rho")

    (RESULTS / "law2_wordnet_summary.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwrote law2_wordnet_summary.json -> {RESULTS}")


if __name__ == "__main__":
    main()
