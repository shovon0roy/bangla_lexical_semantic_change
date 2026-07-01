"""Stage 7: qualitative case studies — how specific words shifted, shown via their
nearest neighbours in the early vs late era, plus a deterministic 2D drift figure.

For each (word, eraA, eraB): align eraB into eraA's space (Procrustes), then list the
word's top-K neighbours in each era (the meaning shift), and plot word+neighbours from
both eras together via PCA (deterministic; arrow = the word's movement).

Words are stemmed to match the corpus. Deterministic (PYTHONHASHSEED=0, PCA via SVD).

Output: results_v2/case_studies.md  +  results_v2/case_<word>_<A>_<B>.png

Run:  .venv/bin/python src/embedding/case_studies_v2.py
"""
import os
import sys
if os.environ.get("PYTHONHASHSEED") != "0":
    os.environ["PYTHONHASHSEED"] = "0"
    os.execv(sys.executable, [sys.executable] + sys.argv)

import contextlib  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
from gensim.models import Word2Vec, KeyedVectors  # noqa: E402
from scipy.linalg import orthogonal_procrustes  # noqa: E402
from bangla_stemmer.stemmer import stemmer  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
MODELS = ROOT / "models_v2"
OUT = ROOT / "results_v2"
K = 10

# (display word, earlier era, later era)
CASES = [
    ("ডিজিটাল", "1990_2010", "2010_2025"),
    ("কুল", "1990_2010", "2010_2025"),
    ("গাজা", "1990_2010", "2010_2025"),
    ("গুম", "1990_2010", "2010_2025"),
    ("স্বাধীনতা", "1950_1970", "2010_2025"),
    ("সাবেক", "1950_1970", "2010_2025"),
    ("মাউস", "1990_2010", "2010_2025"),
    ("নেটওয়ার্ক", "1990_2010", "2010_2025"),
]

_dn = open(os.devnull, "w")
_S = stemmer.BanglaStemmer()


def stem(w):
    with contextlib.redirect_stdout(_dn):
        try:
            return _S.stem(w) or w
        except Exception:
            return w


_cache = {}


def kv(era):
    if era not in _cache:
        _cache[era] = Word2Vec.load(str(MODELS / f"word2vec_{era}.model")).wv
    return _cache[era]


def align(kv1, kv2, n=5000):
    shared = [w for w in kv1.key_to_index if w in kv2.key_to_index]
    shared.sort(key=lambda w: kv1.get_vecattr(w, "count"), reverse=True)
    anc = shared[:n]
    R, _ = orthogonal_procrustes(np.array([kv2[w] for w in anc]), np.array([kv1[w] for w in anc]))
    out = KeyedVectors(vector_size=kv2.vector_size)
    out.add_vectors(list(kv2.index_to_key), kv2.vectors @ R)
    out.fill_norms()
    return out


def pca2(X):
    Xc = X - X.mean(0)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    return Xc @ Vt[:2].T


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib import font_manager
        # try to use a Bengali-capable font if present
        for cand in ["Noto Sans Bengali", "Kalpurush", "Siyam Rupali", "Lohit Bengali"]:
            if any(cand in f.name for f in font_manager.fontManager.ttflist):
                plt.rcParams["font.family"] = cand
                break
        have_plt = True
    except Exception as e:
        have_plt = False
        print(f"(figures off: {e})")

    md = ["# Case studies — neighbour shift (v2 corpus)\n"]
    for word, a, b in CASES:
        sw = stem(word)
        kva, kvb = kv(a), kv(b)
        if sw not in kva or sw not in kvb:
            where = []
            if sw not in kva:
                where.append(f"absent/rare in {a}")
            if sw not in kvb:
                where.append(f"absent/rare in {b}")
            md.append(f"## {word}  (→{sw})\n_{'; '.join(where)} — likely an EMERGENT word (see n-gram prong)._\n")
            print(f"  {word:12} -> skipped ({'; '.join(where)})")
            continue
        kvb_al = align(kva, kvb)
        na = [w for w, _ in kva.most_similar(sw, topn=K)]
        nb = [w for w, _ in kvb_al.most_similar(sw, topn=K)]
        ca, cb = kva.get_vecattr(sw, "count"), kvb.get_vecattr(sw, "count")
        md.append(f"## {word}  (→{sw})   [{a} freq={ca} · {b} freq={cb}]\n")
        md.append(f"- **{a}**: {', '.join(na)}")
        md.append(f"- **{b}**: {', '.join(nb)}\n")
        print(f"  {word:12} [{a}] {', '.join(na[:6])}")
        print(f"  {'':12} [{b}] {', '.join(nb[:6])}")

        if have_plt:
            words_a = [sw] + na
            words_b = [sw] + nb
            X = np.array([kva[w] for w in words_a] + [kvb_al[w] for w in words_b])
            P = pca2(X)
            na_n = len(words_a)
            fig, ax = plt.subplots(figsize=(9, 7))
            ax.scatter(P[:na_n, 0], P[:na_n, 1], c="#1f77b4", s=30, label=a)
            ax.scatter(P[na_n:, 0], P[na_n:, 1], c="#d62728", s=30, label=b)
            for i, w in enumerate(words_a):
                ax.annotate(w, (P[i, 0], P[i, 1]), color="#1f77b4", fontsize=9)
            for i, w in enumerate(words_b):
                ax.annotate(w, (P[na_n + i, 0], P[na_n + i, 1]), color="#d62728", fontsize=9)
            ax.annotate("", xy=(P[na_n, 0], P[na_n, 1]), xytext=(P[0, 0], P[0, 1]),
                        arrowprops=dict(arrowstyle="->", color="black", lw=1.5))
            ax.set_title(f"{word}: neighbourhood drift {a} → {b}")
            ax.legend(); fig.tight_layout()
            fig.savefig(OUT / f"case_{sw}_{a}_{b}.png", dpi=150); plt.close(fig)

    (OUT / "case_studies.md").write_text("\n".join(md), encoding="utf-8")
    print(f"\nwrote case_studies.md + figures -> {OUT}")


if __name__ == "__main__":
    main()
