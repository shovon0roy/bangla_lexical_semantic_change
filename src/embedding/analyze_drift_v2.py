"""Stage 5: align era embeddings (Orthogonal Procrustes) and rank semantic change.

For each era pair (earlier -> later):
  * align the later model into the earlier model's space using Procrustes on the
    top-N most frequent SHARED words (anchors)
  * for every shared word frequent enough in BOTH eras, compute
        delta_cos        = 1 - cos(vec_earlier, vec_later_aligned)      # self-similarity drop
        neighbor_jaccard = |N50_earlier ∩ N50_later| / |N50_earlier ∪ N50_later|
        ChangeScore      = delta_cos/2 + (1 - neighbor_jaccard)         # paper Eq. 4
  * rank words by ChangeScore (descending = most changed)

Deterministic (Procrustes + sorted output; re-execs with PYTHONHASHSEED=0 so tie order
is stable). Outputs go to results_v2/.

Input : models_v2/word2vec_<era>.model
Output: results_v2/drift_<A>_<B>.csv      (word, count_a, count_b, delta_cos, neighbor_jaccard, change_score)
        results_v2/top_changed_<A>_<B>.csv / top_stable_<A>_<B>.csv
        results_v2/summary.json

Run:  .venv/bin/python src/embedding/analyze_drift_v2.py   # --anchors 5000 --min-freq 20 --topk 50
"""
import os
import sys
if os.environ.get("PYTHONHASHSEED") != "0":
    os.environ["PYTHONHASHSEED"] = "0"
    os.execv(sys.executable, [sys.executable] + sys.argv)

import argparse  # noqa: E402
import csv  # noqa: E402
import json  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
from gensim.models import Word2Vec, KeyedVectors  # noqa: E402
from scipy.linalg import orthogonal_procrustes  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
MODELS = ROOT / "models_v2"
OUT = ROOT / "results_v2"
ERAS = ["pre_1950", "1950_1970", "1970_1990", "1990_2010", "2010_2025"]
# consecutive pairs + the full literary span (paper's 1950-vs-2010 comparison)
PAIRS = [(ERAS[i], ERAS[i + 1]) for i in range(len(ERAS) - 1)] + [("1950_1970", "2010_2025")]


def align(kv1, kv2, n_anchor):
    """Rotate kv2 into kv1's space via Procrustes on top-n shared words (by kv1 freq)."""
    shared = [w for w in kv1.key_to_index if w in kv2.key_to_index]
    shared.sort(key=lambda w: kv1.get_vecattr(w, "count"), reverse=True)
    anchors = shared[:n_anchor]
    A = np.array([kv1[w] for w in anchors])
    B = np.array([kv2[w] for w in anchors])
    R, _ = orthogonal_procrustes(B, A)
    aligned = KeyedVectors(vector_size=kv2.vector_size)
    aligned.add_vectors(list(kv2.index_to_key), kv2.vectors @ R)
    # carry counts over so neighbor search / freq filtering still works
    for w in kv2.index_to_key:
        aligned.set_vecattr(w, "count", kv2.get_vecattr(w, "count"))
    aligned.fill_norms()
    return aligned, len(anchors)


def drift(kv1, kv2a, min_freq, topk):
    shared = [w for w in kv1.key_to_index if w in kv2a.key_to_index
              and kv1.get_vecattr(w, "count") >= min_freq
              and kv2a.get_vecattr(w, "count") >= min_freq]
    rows = []
    for i, w in enumerate(shared):
        if (i + 1) % 2000 == 0:
            print(f"    scored {i+1}/{len(shared)} ...", file=sys.stderr)
        v1, v2 = kv1[w], kv2a[w]
        dcos = 1.0 - float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
        n1 = {x for x, _ in kv1.most_similar(w, topn=topk)}
        n2 = {x for x, _ in kv2a.most_similar(w, topn=topk)}
        union = n1 | n2
        jac = len(n1 & n2) / len(union) if union else 0.0
        rows.append((w, int(kv1.get_vecattr(w, "count")), int(kv2a.get_vecattr(w, "count")),
                     round(dcos, 5), round(jac, 5), round(dcos / 2 + (1 - jac), 5)))
    # sort by change_score desc, tie-break by word for determinism
    rows.sort(key=lambda r: (-r[5], r[0]))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--anchors", type=int, default=5000)
    ap.add_argument("--min-freq", type=int, default=20)
    ap.add_argument("--topk", type=int, default=50)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    cache = {}

    def kv(era):
        if era not in cache:
            cache[era] = Word2Vec.load(str(MODELS / f"word2vec_{era}.model")).wv
        return cache[era]

    summary = {"params": vars(args), "pairs": {}}
    for a, b in PAIRS:
        print(f"\n=== {a}  ->  {b} ===")
        kv1, kv2 = kv(a), kv(b)
        kv2a, n_anchor = align(kv1, kv2, args.anchors)
        rows = drift(kv1, kv2a, args.min_freq, args.topk)
        tag = f"{a}_{b}"
        hdr = ["word", "count_a", "count_b", "delta_cos", "neighbor_jaccard", "change_score"]
        with open(OUT / f"drift_{tag}.csv", "w", encoding="utf-8-sig", newline="") as f:
            w = csv.writer(f); w.writerow(hdr); w.writerows(rows)
        with open(OUT / f"top_changed_{tag}.csv", "w", encoding="utf-8-sig", newline="") as f:
            w = csv.writer(f); w.writerow(hdr); w.writerows(rows[:50])
        with open(OUT / f"top_stable_{tag}.csv", "w", encoding="utf-8-sig", newline="") as f:
            w = csv.writer(f); w.writerow(hdr); w.writerows(rows[::-1][:50])
        summary["pairs"][tag] = {"anchors": n_anchor, "scored_words": len(rows),
                                 "mean_change": round(float(np.mean([r[5] for r in rows])), 4)}
        print(f"  anchors={n_anchor:,}  scored={len(rows):,}  mean_change={summary['pairs'][tag]['mean_change']}")
        print(f"  TOP changed: {', '.join(r[0] for r in rows[:12])}")
        print(f"  most stable: {', '.join(r[0] for r in rows[::-1][:12])}")

    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwrote drift CSVs + summary.json -> {OUT}")


if __name__ == "__main__":
    main()
