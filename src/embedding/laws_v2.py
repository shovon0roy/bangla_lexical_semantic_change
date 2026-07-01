"""Stage 6: test the two diachronic 'laws' on the v2 corpus (full span 1950-70 -> 2010-25).

Law 1 (Law of Conformity): higher-frequency words change LESS.
    Spearman(log10 early-era frequency, ChangeScore)  -> expect negative rho.
Law 2 (Law of Innovation): more polysemous words change MORE.
    polysemy proxy = neighbour dispersion = 1 - mean pairwise cosine among the top-50
    neighbours in the EARLY model. Spearman(dispersion, ChangeScore) -> expect positive
    rho (the original paper found this NULL).

Deterministic (re-execs with PYTHONHASHSEED=0). Uses results_v2/drift_1950_1970_2010_2025.csv
(which already carries count_a = early-era frequency) + models_v2/word2vec_1950_1970.model.

Output: results_v2/laws_summary.json, results_v2/laws_data.csv,
        results_v2/figure_law1_frequency_vs_change.png / figure_law2_polysemy_vs_change.png

Run:  .venv/bin/python src/embedding/laws_v2.py
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
from scipy.stats import spearmanr  # noqa: E402
from gensim.models import Word2Vec  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results_v2"
MODELS = ROOT / "models_v2"
DRIFT = RESULTS / "drift_1950_1970_2010_2025.csv"
EARLY_MODEL = MODELS / "word2vec_1950_1970.model"
K = 50


def dispersion(kv, word, k=K):
    try:
        nbrs = [w for w, _ in kv.most_similar(word, topn=k)]
    except KeyError:
        return None
    if len(nbrs) < 2:
        return None
    V = np.array([kv[w] for w in nbrs])
    V = V / np.linalg.norm(V, axis=1, keepdims=True)
    sim = V @ V.T
    iu = np.triu_indices(len(sim), k=1)
    return float(1.0 - sim[iu].mean())


def main():
    rows = list(csv.DictReader(open(DRIFT, encoding="utf-8-sig")))
    kv = Word2Vec.load(str(EARLY_MODEL)).wv
    print(f"words: {len(rows)}  |  early model vocab: {len(kv):,}\n")

    logf, disp, change, words = [], [], [], []
    for i, r in enumerate(rows):
        if (i + 1) % 2000 == 0:
            print(f"  dispersion {i+1}/{len(rows)} ...", file=sys.stderr)
        w = r["word"]
        ca = int(r["count_a"])
        cs = float(r["change_score"])
        d = dispersion(kv, w)
        if d is None or ca <= 0:
            continue
        words.append(w); logf.append(np.log10(ca)); disp.append(d); change.append(cs)

    logf, disp, change = np.array(logf), np.array(disp), np.array(change)

    rho1, p1 = spearmanr(logf, change)
    rho2, p2 = spearmanr(disp, change)
    res = {
        "pair": "1950_1970_vs_2010_2025", "n_words": len(words),
        "law1_frequency": {"spearman_rho": round(float(rho1), 4), "p_value": float(p1),
                           "significant": bool(p1 < 0.05), "direction": "negative=conformity"},
        "law2_polysemy": {"spearman_rho": round(float(rho2), 4), "p_value": float(p2),
                          "significant": bool(p2 < 0.05), "direction": "positive=innovation"},
    }
    print("=== LAW 1  (frequency vs change) ===")
    print(f"  Spearman rho = {rho1:+.4f}  p = {p1:.3g}  -> "
          f"{'SIGNIFICANT' if p1 < 0.05 else 'n.s.'}  ({'CONFORMITY confirmed' if (rho1 < 0 and p1 < 0.05) else 'check'})")
    print("=== LAW 2  (polysemy vs change) ===")
    print(f"  Spearman rho = {rho2:+.4f}  p = {p2:.3g}  -> "
          f"{'SIGNIFICANT' if p2 < 0.05 else 'n.s.'}  ({'INNOVATION confirmed' if (rho2 > 0 and p2 < 0.05) else 'NULL / not confirmed'})")

    (RESULTS / "laws_summary.json").write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    with open(RESULTS / "laws_data.csv", "w", encoding="utf-8-sig", newline="") as f:
        wr = csv.writer(f); wr.writerow(["word", "log_frequency", "polysemy_dispersion", "change_score"])
        for i in range(len(words)):
            wr.writerow([words[i], round(float(logf[i]), 4), round(float(disp[i]), 4), round(float(change[i]), 4)])

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        for x, xlab, fn, rho, p in [
            (logf, "log10 frequency (1950-70)", "figure_law1_frequency_vs_change.png", rho1, p1),
            (disp, "polysemy proxy (neighbour dispersion)", "figure_law2_polysemy_vs_change.png", rho2, p2)]:
            fig, ax = plt.subplots(figsize=(7, 5))
            ax.scatter(x, change, s=4, alpha=0.25, color="#3b6")
            m, b = np.polyfit(x, change, 1)
            xs = np.linspace(x.min(), x.max(), 50)
            ax.plot(xs, m * xs + b, color="red", lw=2)
            ax.set_xlabel(xlab); ax.set_ylabel("ChangeScore (1950-2025)")
            ax.set_title(f"rho={rho:+.3f}, p={p:.2g}")
            fig.tight_layout(); fig.savefig(RESULTS / fn, dpi=150); plt.close(fig)
        print("\n  saved scatter plots (law1/law2 png)")
    except Exception as e:
        print(f"\n  (figures skipped: {e})")

    print(f"\nwrote laws_summary.json + laws_data.csv -> {RESULTS}")


if __name__ == "__main__":
    main()
