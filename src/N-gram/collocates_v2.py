"""PMI collocates per era for the case-study words, on the v2 corpus.
Feeds the collocate heatmaps (কুল, স্বাধীনতা) and the ডিজিটাল drift discussion.

Same PMI as the original collocates.py: window 5, min co-occurrence 3,
pmi = log2( p(collocate | target) / p(collocate) ).

Output: results_v2/collocates_<word>.json  = { era: { collocate: pmi, ... }, ... }

Run:  .venv/bin/python src/N-gram/collocates_v2.py
"""
import contextlib
import json
import math
import os
from collections import Counter
from pathlib import Path

from bangla_stemmer.stemmer import stemmer

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data_cleaned_v2"
OUT = ROOT / "results_v2"
ERAS = ["pre_1950", "1950_1970", "1970_1990", "1990_2010", "2010_2025"]
WINDOW, MIN_COOCC, TOPK = 5, 3, 12
TARGETS = ["দারুণ", "স্বাধীনতা", "ডিজিটাল"]

_dn = open(os.devnull, "w")
_S = stemmer.BanglaStemmer()


def stem(w):
    with contextlib.redirect_stdout(_dn):
        try:
            return _S.stem(w) or w
        except Exception:
            return w


def sentences(path):
    out, cur = [], []
    for tok in path.read_text(encoding="utf-8").split():
        if tok == "<s>" or tok == "<s/>":
            if cur:
                out.append(cur); cur = []
        else:
            cur.append(tok)
    if cur:
        out.append(cur)
    return out


def pmi_for(target, sents, counts, ntok):
    occ = 0
    co = Counter()
    for s in sents:
        for i, t in enumerate(s):
            if t == target:
                occ += 1
                lo, hi = max(0, i - WINDOW), min(len(s), i + WINDOW + 1)
                for c in s[lo:i] + s[i + 1:hi]:
                    if c != target:
                        co[c] += 1
    if occ == 0:
        return {}
    windows = occ * WINDOW * 2
    out = {}
    for c, n in co.items():
        if n < MIN_COOCC:
            continue
        p_ct = n / windows
        p_c = counts[c] / ntok
        if p_ct > 0 and p_c > 0:
            out[c] = math.log2(p_ct / p_c)
    return out


def main():
    targets = {t: stem(t) for t in TARGETS}
    print("stemmed targets:", targets)
    per_era_pmi = {t: {} for t in TARGETS}
    for era in ERAS:
        fp = DATA / f"{era}.txt"
        if not fp.exists():
            continue
        sents = sentences(fp)
        flat = [w for s in sents for w in s]
        counts = Counter(flat); ntok = len(flat)
        for disp, st in targets.items():
            per_era_pmi[disp][era] = pmi_for(st, sents, counts, ntok)
        print(f"  {era}: done")

    for disp in TARGETS:
        # keep the union of each era's top-K collocates (so the heatmap has stable rows)
        keep = set()
        for era in ERAS:
            top = sorted(per_era_pmi[disp].get(era, {}).items(), key=lambda x: -x[1])[:TOPK]
            keep.update(w for w, _ in top)
        table = {era: {w: round(per_era_pmi[disp].get(era, {}).get(w, 0.0), 3) for w in sorted(keep)}
                 for era in ERAS}
        (OUT / f"collocates_{disp}.json").write_text(json.dumps(table, ensure_ascii=False, indent=1),
                                                     encoding="utf-8")
        print(f"wrote collocates_{disp}.json  ({len(keep)} collocates)")


if __name__ == "__main__":
    main()
