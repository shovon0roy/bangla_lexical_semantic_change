"""Regenerate every data figure in the paper from the v2 results, into
book/figures/ under the SAME filenames the manuscript already \\includegraphics.

Figures:
  changescore_distribution.jpg  (Fig 3)  histogram of ChangeScores
  frequency_trends.jpg          (Fig 4)  per-million frequency of 4 showcase words
  drift_digital.jpg             (Fig 5)  PCA neighbourhood drift of ডিজিটাল
  heatmap_darun.jpg             (Fig 6)  PMI collocate heatmap of দারুণ
  heatmap_swadhinata.jpg        (Fig 7)  PMI collocate heatmap of স্বাধীনতা
  law_frequency.jpg / law_polysemy.jpg (Fig 8)  frequency vs change; sense-count vs change

Run:  PYTHONHASHSEED=0 .venv/bin/python src/make_figures_v2.py
"""
import os
import sys
if os.environ.get("PYTHONHASHSEED") != "0":
    os.environ["PYTHONHASHSEED"] = "0"
    os.execv(sys.executable, [sys.executable] + sys.argv)

import contextlib
import csv
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from scipy.linalg import orthogonal_procrustes
from gensim.models import Word2Vec, KeyedVectors
from bangla_stemmer.stemmer import stemmer

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results_v2"
MODELS = ROOT / "models_v2"
DATA = ROOT / "data_cleaned_v2"
FIG = ROOT / "book" / "figures"
ERAS = ["pre_1950", "1950_1970", "1970_1990", "1990_2010", "2010_2025"]
ERA_LBL = ["pre1950", "1950-70", "1970-90", "1990-2010", "2010-25"]

plt.rcParams["font.family"] = ["DejaVu Sans", "Kalpurush"]   # Latin + Bengali fallback
plt.rcParams["axes.unicode_minus"] = False
GREEN = "#2a8"
random.seed(0); np.random.seed(0)

_dn = open(os.devnull, "w")
_S = stemmer.BanglaStemmer()
def stem(w):
    with contextlib.redirect_stdout(_dn):
        try: return _S.stem(w) or w
        except Exception: return w


def save(fig, name):
    fig.savefig(FIG / name, dpi=600, bbox_inches="tight")   # ACM final: >=600 dpi
    plt.close(fig)
    print("  wrote", name)


# ---- Fig 3: ChangeScore distribution ---------------------------------------
def fig_distribution():
    rows = list(csv.DictReader(open(RES / "drift_1950_1970_2010_2025.csv", encoding="utf-8-sig")))
    cs = np.array([float(r["change_score"]) for r in rows])
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.hist(cs, bins=80, color=GREEN, alpha=0.85)
    ax.axvline(cs.mean(), color="red", ls="--", lw=2, label=f"mean = {cs.mean():.2f}")
    ax.set_xlabel("ChangeScore (1950–70 vs 2010–25)"); ax.set_ylabel("number of words")
    ax.legend(); save(fig, "changescore_distribution.jpg")


# ---- Fig 4: frequency trends -----------------------------------------------
def fig_freqtrends():
    words = ["ডিজিটাল", "ভাইরাল", "দারুণ", "অর্বাচীন"]
    sw = {w: stem(w) for w in words}
    pm = {w: [] for w in words}
    for era in ERAS:
        toks = [t for t in (DATA / f"{era}.txt").read_text(encoding="utf-8").split() if t not in ("<s>", "<s/>")]
        n = len(toks); c = Counter(toks)
        for w in words:
            pm[w].append(c[sw[w]] / n * 1e6 if n else 0)
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    xs = range(len(ERAS))
    markers = ["o", "s", "^", "D"]           # distinct shapes + line styles so the four
    styles = ["-", "--", "-.", ":"]          # lines stay distinguishable in greyscale (ACM)
    for w, mk, ls in zip(words, markers, styles):
        ax.plot(xs, pm[w], marker=mk, linestyle=ls, lw=2, markersize=7, label=w)
    ax.set_xticks(list(xs)); ax.set_xticklabels(ERA_LBL, rotation=15)
    ax.set_ylabel("frequency per million"); ax.legend()
    save(fig, "frequency_trends.jpg")


# ---- embedding alignment helper --------------------------------------------
def kv(era):
    return Word2Vec.load(str(MODELS / f"word2vec_{era}.model")).wv
def align(k1, k2, n=5000):
    shared = [w for w in k1.key_to_index if w in k2.key_to_index]
    shared.sort(key=lambda w: k1.get_vecattr(w, "count"), reverse=True)
    a = shared[:n]
    R, _ = orthogonal_procrustes(np.array([k2[w] for w in a]), np.array([k1[w] for w in a]))
    out = KeyedVectors(vector_size=k2.vector_size)
    out.add_vectors(list(k2.index_to_key), k2.vectors @ R); out.fill_norms()
    return out
def pca2(X):
    Xc = X - X.mean(0); U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    return Xc @ Vt[:2].T


# ---- Fig 5: drift of ডিজিটাল (UMAP, star + arrow, era-coloured) -------------
def fig_drift_digital():
    import umap
    w = stem("ডিজিটাল"); a, b = "1990_2010", "2010_2025"
    k1, k2 = kv(a), kv(b); k2a = align(k1, k2)
    if w not in k1 or w not in k2a:
        print("  ডিজিটাল missing, skip drift"); return
    K = 15
    na = [w] + [x for x, _ in k1.most_similar(w, topn=K)]
    nb = [w] + [x for x, _ in k2a.most_similar(w, topn=K)]
    X = np.vstack([np.array([k1[x] for x in na]), np.array([k2a[x] for x in nb])])
    P = umap.UMAP(n_components=2, random_state=42, n_neighbors=12, min_dist=0.6).fit_transform(X)
    m = len(na)
    BLUE, RED = "#3b46c4", "#e23b34"
    fig, ax = plt.subplots(figsize=(11, 7.5))
    ax.scatter(P[1:m, 0], P[1:m, 1], c=BLUE, s=46, marker="o", label="1990–2010", zorder=3)
    ax.scatter(P[m + 1:, 0], P[m + 1:, 1], c=RED, s=64, marker="s", label="2010–25", zorder=3)
    ax.scatter(P[0, 0], P[0, 1], c=BLUE, s=520, marker="*", edgecolor="white", lw=1, zorder=5)
    ax.scatter(P[m, 0], P[m, 1], c=RED, s=520, marker="*", edgecolor="white", lw=1, zorder=5)
    for i, x in enumerate(na):
        ax.annotate(x, (P[i, 0], P[i, 1]), color=BLUE, fontsize=10, ha="center",
                    xytext=(0, 7), textcoords="offset points")
    for i, x in enumerate(nb):
        ax.annotate(x, (P[m + i, 0], P[m + i, 1]), color=RED, fontsize=10, ha="center",
                    xytext=(0, 7), textcoords="offset points")
    ax.annotate("", xy=(P[m, 0], P[m, 1]), xytext=(P[0, 0], P[0, 1]),
                arrowprops=dict(arrowstyle="-|>", color="black", lw=2), zorder=4)
    ax.grid(True, color="#dde6f0", lw=0.8); ax.set_axisbelow(True)
    ax.set_xlabel("UMAP dimension 1"); ax.set_ylabel("UMAP dimension 2")
    ax.legend(loc="upper right", framealpha=0.9)
    save(fig, "drift_digital.jpg")


# ---- Fig 6/7: collocate heatmaps -------------------------------------------
def fig_heatmap(disp, fname, rows=None):
    f = RES / f"collocates_{disp}.json"
    if not f.exists():
        print(f"  {f.name} missing, skip"); return
    table = json.loads(f.read_text(encoding="utf-8"))
    if rows is not None:                                # curated, interpretable collocates
        allc = [c for c in rows if c in table[ERAS[0]]]
        Mall = np.array([[table[e].get(c, 0.0) for e in ERAS] for c in allc])
    else:
        allc = list(table[ERAS[0]].keys())
        Mall = np.array([[table[e].get(c, 0.0) for e in ERAS] for c in allc])
        keep = np.argsort(-Mall.var(axis=1))[:16]      # the 16 most era-varying collocates
        allc = [allc[i] for i in keep]; Mall = Mall[keep]
    order = np.argsort(Mall.argmax(axis=1))             # sort by which era they peak in
    cols = [allc[i] for i in order]; M = Mall[order]
    fig, ax = plt.subplots(figsize=(7, max(4, 0.34 * len(cols))))
    im = ax.imshow(M, aspect="auto", cmap="YlGnBu")
    ax.set_xticks(range(len(ERAS))); ax.set_xticklabels(ERA_LBL, rotation=20)
    ax.set_yticks(range(len(cols))); ax.set_yticklabels(cols)
    fig.colorbar(im, ax=ax, label="PPMI")
    save(fig, fname)


# ---- Fig 8: the two laws ---------------------------------------------------
def components(words, kv_, edge=0.45):
    V = np.array([kv_[w] for w in words]); V = V / np.linalg.norm(V, axis=1, keepdims=True)
    A = (V @ V.T) >= edge; n = len(words); seen = [False] * n; comp = 0
    for i in range(n):
        if seen[i]: continue
        comp += 1; st = [i]; seen[i] = True
        while st:
            u = st.pop()
            for v in range(n):
                if A[u, v] and not seen[v]: seen[v] = True; st.append(v)
    return comp

def fig_laws():
    data = list(csv.DictReader(open(RES / "laws_data.csv", encoding="utf-8-sig")))
    logf = np.array([float(r["log_frequency"]) for r in data])
    chg = np.array([float(r["change_score"]) for r in data])
    r1, p1 = spearmanr(logf, chg)
    fig, ax = plt.subplots(figsize=(6, 4.6))
    ax.scatter(logf, chg, s=5, alpha=0.18, color=GREEN)
    m, b = np.polyfit(logf, chg, 1); xs = np.linspace(logf.min(), logf.max(), 50)
    ax.plot(xs, m * xs + b, color="red", lw=2)
    ax.set_xlabel("log$_{10}$ frequency (early era)"); ax.set_ylabel("ChangeScore")
    ax.set_title(f"Law of Frequency: ρ = {r1:+.3f}, p < 0.001")
    save(fig, "law_frequency.jpg")

    # Law of Polysemy via a proper sense-count (connected components of the neighbour
    # graph at a threshold that actually splits senses). Box plot by sense-count.
    EDGE = 0.60
    kv1 = kv("1950_1970")
    samp = [r for r in data if r["word"] in kv1]   # full shared vocabulary (no sampling),
    sc, ch = [], []                                 # so the figure matches law2_polysemy_v2
    for r in samp:
        nb = [x for x, _ in kv1.most_similar(r["word"], topn=50)]
        if len(nb) < 5:
            continue
        sc.append(components(nb, kv1, EDGE)); ch.append(float(r["change_score"]))
    sc, ch = np.array(sc), np.array(ch)
    r2, p2 = spearmanr(sc, ch)
    groups = [ch[sc == 1], ch[sc == 2], ch[sc >= 3]]
    fig, ax = plt.subplots(figsize=(6, 4.6))
    bp = ax.boxplot(groups, tick_labels=["1", "2", "≥3"], showfliers=False, patch_artist=True)
    for box in bp["boxes"]:
        box.set(facecolor="#cde", alpha=0.8)
    ax.set_xlabel("number of sense clusters (polysemy)"); ax.set_ylabel("ChangeScore")
    ax.set_title(f"Law of Polysemy: ρ = {r2:+.3f}, p = {p2:.3f} (not confirmed)")
    save(fig, "law_polysemy.jpg")
    print(f"  [law numbers] frequency ρ={r1:+.4f} p={p1:.2e} | polysemy(sense) ρ={r2:+.4f} p={p2:.4f}")


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    print("generating v2 figures ->", FIG)
    fig_distribution()
    fig_freqtrends()
    fig_drift_digital()
    # curated collocates for দারুণ: negative-affect (early) -> appearance -> sports (late)
    darun_rows = ["আতঙ্", "ঘৃণায়", "যন্ত্রণায়", "অপমানে", "দুর্ভাবনা", "অস্বস্তি", "লোকসান",
                  "অ্যাট্রাকটিভ", "হ্যান্ডসাম", "স্মার্ট", "ইন্টারেস্টিং",
                  "থ্রিলিং", "নৈপুণ্য", "ফিনিশিংয়ে", "ভলি"]
    fig_heatmap("দারুণ", "heatmap_darun.jpg", rows=darun_rows)
    # curated collocates for স্বাধীনতা: anticolonial (early) -> 1971 commemoration -> rights
    swadhinata_rows = ["পরাধীনতা", "ব্রিটেনের", "ঐক্যবদ্ধ", "শহিদ", "অবাধ", "অর্থনৈতিক",
                       "আন্দোলন", "সংগ্রাম", "সংগ্রামী", "বিসর্জন", "অর্জন", "দিবস", "দিবসে",
                       "মুক্তিযুদ্ধে", "মতপ্রকাশ", "সার্বভৌমত্ব"]
    fig_heatmap("স্বাধীনতা", "heatmap_swadhinata.jpg", rows=swadhinata_rows)
    fig_laws()


if __name__ == "__main__":
    main()
