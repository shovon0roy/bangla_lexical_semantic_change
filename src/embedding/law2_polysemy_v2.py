"""Stage 6c: settle Law 2 — does 'more polysemous -> more change' hold, or is the raw
negative result an artifact of the dispersion proxy measuring GENERICNESS, not polysemy?

Three polysemy/structure proxies on the EARLY model, each vs ChangeScore (Spearman),
plus frequency-partialled correlations (control for the conformity law):

  1. dispersion      = 1 - mean pairwise cosine among top-K neighbours
                       (high = diffuse/generic; the original proxy)
  2. sense_clusters  = # connected components of the neighbour graph (edge if
                       cosine>EDGE). More components = more distinct senses = polysemy.
  3. max_sim         = similarity to the single nearest neighbour
                       (high = sits in a tight, well-defined region)

If sense_clusters and dispersion disagree in sign vs change, the negative 'law' reflects
genericness, not polysemy — and the Law of Innovation is better tested by sense_clusters.

Deterministic. Reuses results_v2/laws_data.csv (change, logf) + early model.

Output: results_v2/law2_polysemy_summary.json (+ console)

Run:  .venv/bin/python src/embedding/law2_polysemy_v2.py
"""
import os
import sys
if os.environ.get("PYTHONHASHSEED") != "0":
    os.environ["PYTHONHASHSEED"] = "0"
    os.execv(sys.executable, [sys.executable] + sys.argv)

import csv  # noqa: E402
import json  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
from scipy.stats import spearmanr, rankdata  # noqa: E402
from gensim.models import Word2Vec  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results_v2"
MODELS = ROOT / "models_v2"
EARLY = "1950_1970"
K = 50
EDGE = 0.60      # neighbour-graph edge threshold (cosine). At working thresholds (0.55-0.65)
                 # the sense-counts actually vary and give sense_clusters rho ~ -0.06, p < 0.001
                 # (the value reported in the paper). A threshold that is too low (e.g. 0.45)
                 # collapses almost every word into ONE component, so the sense-count is nearly
                 # constant and rho ~ 0 with p ~ 0.9 — a degenerate artifact, not a real null.


def components(words, kv, edge=EDGE):
    """# connected components among a word's neighbours (proxy for # senses)."""
    n = len(words)
    if n == 0:
        return 0
    V = np.array([kv[w] for w in words])
    V = V / np.linalg.norm(V, axis=1, keepdims=True)
    A = (V @ V.T) >= edge
    seen = [False] * n
    comp = 0
    for i in range(n):
        if seen[i]:
            continue
        comp += 1
        stack = [i]
        seen[i] = True
        while stack:
            u = stack.pop()
            for v in range(n):
                if A[u, v] and not seen[v]:
                    seen[v] = True
                    stack.append(v)
    return comp


def partial_spearman(x, y, z):
    """Spearman partial correlation of x,y controlling for z (rank-residualised)."""
    rx, ry, rz = rankdata(x), rankdata(y), rankdata(z)
    def resid(a, b):
        b1 = np.vstack([b, np.ones_like(b)]).T
        coef, *_ = np.linalg.lstsq(b1, a, rcond=None)
        return a - b1 @ coef
    ex, ey = resid(rx, rz), resid(ry, rz)
    r = np.corrcoef(ex, ey)[0, 1]
    return float(r)


def main():
    data = {r["word"]: r for r in csv.DictReader(open(RESULTS / "laws_data.csv", encoding="utf-8-sig"))}
    kv = Word2Vec.load(str(MODELS / f"word2vec_{EARLY}.model")).wv

    words, disp, sclu, msim, change, logf = [], [], [], [], [], []
    items = list(data.items())
    for i, (w, d) in enumerate(items):
        if (i + 1) % 2000 == 0:
            print(f"  {i+1}/{len(items)} ...", file=sys.stderr)
        if w not in kv:
            continue
        nb = [x for x, _ in kv.most_similar(w, topn=K)]
        if len(nb) < 5:
            continue
        V = np.array([kv[x] for x in nb]); V = V / np.linalg.norm(V, axis=1, keepdims=True)
        sim = V @ V.T
        iu = np.triu_indices(len(sim), k=1)
        words.append(w)
        disp.append(1 - sim[iu].mean())
        sclu.append(components(nb, kv))
        msim.append(float(max(s for _, s in kv.most_similar(w, topn=1))))
        change.append(float(d["change_score"]))
        logf.append(float(d["log_frequency"]))

    disp, sclu, msim = np.array(disp), np.array(sclu), np.array(msim)
    change, logf = np.array(change), np.array(logf)
    n = len(words)

    out = {"n": n, "early_era": EARLY, "K": K, "edge": EDGE, "proxies": {}}
    print(f"\nwords: {n}\n")
    print(f"{'proxy':16} {'rho vs change':>14} {'p':>10} {'partial|freq':>14}")
    for name, x, exp in [("dispersion", disp, "neg=generic-stable"),
                         ("sense_clusters", sclu, "pos=Law-of-Innovation"),
                         ("max_sim(top1)", msim, "tight region")]:
        rho, p = spearmanr(x, change)
        pr = partial_spearman(x, change, logf)
        out["proxies"][name] = {"spearman_rho": round(float(rho), 4), "p": float(p),
                                "partial_rho_given_freq": round(pr, 4), "note": exp}
        print(f"{name:16} {rho:>+14.4f} {p:>10.1e} {pr:>+14.4f}   ({exp})")

    # also: does sense_clusters track dispersion? (if anti-correlated, they measure diff things)
    rsd, _ = spearmanr(sclu, disp)
    out["sense_vs_dispersion_rho"] = round(float(rsd), 4)
    print(f"\nsense_clusters vs dispersion rho = {rsd:+.4f}")
    print("Read: if sense_clusters->change is POSITIVE while dispersion->change is NEGATIVE,")
    print("the negative 'law' is genericness, and Law of Innovation holds under sense-count.")

    (RESULTS / "law2_polysemy_summary.json").write_text(json.dumps(out, ensure_ascii=False, indent=2),
                                                        encoding="utf-8")
    print(f"\nwrote law2_polysemy_summary.json -> {RESULTS}")


if __name__ == "__main__":
    main()
