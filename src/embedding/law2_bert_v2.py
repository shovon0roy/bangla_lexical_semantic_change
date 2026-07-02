"""Option 3: test the Law of Polysemy with contextual embeddings (BanglaBERT).

Static embeddings average a word's senses into one vector, so they structurally
understate the sense-REBALANCING kind of change the Law of Innovation predicts.
This script measures change at the occurrence level instead:

  1. targets   : a stratified, deterministic sample of the IndoWordNet-matched
                 vocabulary (sense buckets 1/2/3/4+ x frequency tertiles).
  2. contexts  : up to N_CTX raw literary sentences per word per era, sampled from
                 the un-stemmed epub_dataset text of 1950-1970 and 2010-2025
                 (register-matched: literary prose in both eras). Raw tokens are
                 matched to targets through the same bangla_stemmer as the corpus.
  3. vectors   : BanglaBERT (csebuetnlp/banglabert) last-hidden-layer vectors of
                 the target token (mean over its subword pieces).
  4. change    : APD = average pairwise cosine distance between era-A and era-B
                 occurrence vectors (SemEval-2020 standard), which sees a shift in
                 the MIX of senses even when the average stands still; plus a
                 secondary cluster measure, JSD between the two eras' distributions
                 over k-means usage clusters (k=5, fixed for every word).
  5. test      : Spearman(WordNet senses, change), partial given log frequency,
                 with frequency-tertile breakdown. The law predicts positive rho.

Deterministic: seeded reservoir sampling, sorted file order, k-means random_state,
model in eval mode. Intermediates cached in results_v2/bert_cache/.

Run:  .venv/bin/python src/embedding/law2_bert_v2.py
Output: results_v2/law2_bert_summary.json
"""
import contextlib
import gzip
import json
import math
import os
import random
import re
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr, rankdata

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results_v2"
CACHE = RESULTS / "bert_cache"
RAW = ROOT / "corpus_build" / "epub_dataset"
ERA_A, ERA_B = "1950_1970", "2010_2025"
N_CTX = 50            # contexts per word per era
MIN_CTX = 20          # a word needs >= this many in BOTH eras to be scored
PER_CELL = 90         # stratified cap per (sense-bucket x freq-tertile) cell
HALF_WIN = 25         # raw tokens kept either side of the target occurrence
MODEL = "csebuetnlp/banglabert"
KMEANS_K = 5
SEED = 0

nfc = lambda s: unicodedata.normalize("NFC", s)
SENT_SPLIT = re.compile(r"[।?!\n]+")
EDGE_PUNCT = re.compile(r"^[\"'“”‘’,.;:()\[\]{}\-–—…]+|[\"'“”‘’,.;:()\[\]{}\-–—…]+$")
HAS_ASCII = re.compile(r"[A-Za-z0-9]")

_dn = open(os.devnull, "w")
_S = None
_stem_cache = {}


def stem(tok):
    global _S
    if tok in _stem_cache:
        return _stem_cache[tok]
    if _S is None:
        from bangla_stemmer.stemmer import stemmer
        _S = stemmer.BanglaStemmer()
    with contextlib.redirect_stdout(_dn):
        try:
            s = _S.stem(tok) or tok
        except Exception:
            s = tok
    _stem_cache[tok] = s
    return s


def wordnet_senses():
    import pyiwn
    iwn = pyiwn.IndoWordNet(lang=pyiwn.Language.BENGALI)
    sense = {}
    for w in iwn.all_words():
        if "_" not in w:
            try:
                sense[nfc(w)] = len(iwn.synsets(w))
            except Exception:
                pass
    return sense


def pick_targets():
    """Stratified deterministic target sample from the WordNet-matched vocab."""
    import csv
    sense = wordnet_senses()
    rows = list(csv.DictReader(open(RESULTS / "laws_data.csv", encoding="utf-8-sig")))
    matched = [(nfc(r["word"]), sense[nfc(r["word"])], float(r["log_frequency"]))
               for r in rows if nfc(r["word"]) in sense]
    logf = np.array([m[2] for m in matched])
    t1, t2 = np.quantile(logf, [1 / 3, 2 / 3])
    cells = defaultdict(list)
    for (w, s, f) in matched:
        sb = "1" if s == 1 else "2" if s == 2 else "3" if s == 3 else "4+"
        fb = "lo" if f <= t1 else "mid" if f <= t2 else "hi"
        cells[(sb, fb)].append((w, s, f))
    rng = random.Random(SEED)
    targets = {}
    for key in sorted(cells):
        pool = sorted(cells[key])
        rng.shuffle(pool)
        for (w, s, f) in pool[:PER_CELL]:
            targets[w] = {"senses": s, "logf": f}
    print(f"targets: {len(targets)} words across {len(cells)} strata")
    return targets


def extract_contexts(targets):
    """Reservoir-sample up to N_CTX raw sentences per target per era."""
    out = {era: {w: [] for w in targets} for era in (ERA_A, ERA_B)}
    seen = {era: {w: 0 for w in targets} for era in (ERA_A, ERA_B)}
    rng = random.Random(SEED)
    for era in (ERA_A, ERA_B):
        files = sorted((RAW / era).glob("*.txt"))
        print(f"  scanning {era}: {len(files)} files")
        for fi, fp in enumerate(files):
            try:
                text = nfc(fp.read_text(encoding="utf-8", errors="ignore"))
            except Exception:
                continue
            for sent in SENT_SPLIT.split(text):
                raw_toks = sent.split()
                if not (3 <= len(raw_toks) <= 400):
                    continue
                for i, rt in enumerate(raw_toks):
                    tok = EDGE_PUNCT.sub("", rt)
                    if not tok or HAS_ASCII.search(tok):
                        continue
                    st = stem(tok)
                    if st not in out[era]:
                        continue
                    seen[era][st] += 1
                    lo = max(0, i - HALF_WIN)
                    ctx_toks = raw_toks[lo:i + HALF_WIN + 1]
                    rec = (" ".join(ctx_toks), i - lo, tok)
                    bucket = out[era][st]
                    if len(bucket) < N_CTX:
                        bucket.append(rec)
                    else:                                   # reservoir sampling
                        j = rng.randrange(seen[era][st])
                        if j < N_CTX:
                            bucket[j] = rec
            if (fi + 1) % 200 == 0:
                print(f"    {era}: {fi+1}/{len(files)} files", file=sys.stderr)
    return out


def encode(contexts):
    """BanglaBERT vectors for every stored occurrence (mean over target subwords)."""
    import torch
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModel.from_pretrained(MODEL)
    model.eval()
    torch.set_grad_enabled(False)
    vecs = {era: {} for era in contexts}
    items = []
    for era in contexts:
        for w, recs in contexts[era].items():
            for k, (sent, tpos, surface) in enumerate(recs):
                # char span of the target token inside the sentence string
                start = len(" ".join(sent.split()[:tpos])) + (1 if tpos else 0)
                items.append((era, w, k, sent, start, start + len(sent.split()[tpos])))
    print(f"  encoding {len(items):,} occurrences ...")
    B = 32
    for bi in range(0, len(items), B):
        batch = items[bi:bi + B]
        enc = tok([b[3] for b in batch], return_tensors="pt", padding=True,
                  truncation=True, max_length=128, return_offsets_mapping=True)
        offsets = enc.pop("offset_mapping")
        h = model(**enc).last_hidden_state          # [B, T, H]
        for r, (era, w, k, sent, cs, ce) in enumerate(batch):
            om = offsets[r]
            idx = [t for t in range(om.shape[0])
                   if om[t][1] > om[t][0] and om[t][0] < ce and om[t][1] > cs]
            if not idx:
                continue
            v = h[r, idx].mean(dim=0).numpy()
            vecs[era].setdefault(w, []).append(v)
        if (bi // B) % 100 == 0:
            print(f"    {bi:,}/{len(items):,}", file=sys.stderr)
    return {era: {w: np.vstack(v) for w, v in d.items() if len(v) >= MIN_CTX}
            for era, d in vecs.items()}


def apd(A, B, rng, max_pairs=2000, exclude_diag=False):
    """Average pairwise cosine distance between two sets of occurrence vectors.
    With A is B and exclude_diag, this is the WITHIN-era spread of the word's
    occurrence cloud — the synchronic-polysemy baseline that raw between-era APD
    silently absorbs (a polysemous word's cloud is wide even with zero change)."""
    An = A / np.linalg.norm(A, axis=1, keepdims=True)
    Bn = B / np.linalg.norm(B, axis=1, keepdims=True)
    sims = An @ Bn.T
    flat = sims[~np.eye(len(A), len(B), dtype=bool)] if exclude_diag else sims.ravel()
    if flat.size > max_pairs:
        idx = rng.sample(range(flat.size), max_pairs)
        flat = flat[idx]
    return float(1.0 - flat.mean())


def cluster_jsd(A, B):
    from sklearn.cluster import KMeans
    X = np.vstack([A, B])
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    k = min(KMEANS_K, len(X) - 1)
    lab = KMeans(n_clusters=k, n_init=4, random_state=SEED).fit_predict(X)
    pa = np.bincount(lab[:len(A)], minlength=k).astype(float)
    pb = np.bincount(lab[len(A):], minlength=k).astype(float)
    pa /= pa.sum()
    pb /= pb.sum()
    m = 0.5 * (pa + pb)
    def kl(p, q):
        mask = p > 0
        return float(np.sum(p[mask] * np.log2(p[mask] / q[mask])))
    return 0.5 * kl(pa, m) + 0.5 * kl(pb, m)


def partial_spearman(x, y, z):
    rx, ry, rz = rankdata(x), rankdata(y), rankdata(z)
    def resid(a, b):
        b1 = np.vstack([b, np.ones_like(b)]).T
        coef, *_ = np.linalg.lstsq(b1, a, rcond=None)
        return a - b1 @ coef
    return float(np.corrcoef(resid(rx, rz), resid(ry, rz))[0, 1])


def report(name, senses, change, logf):
    rho, p = spearmanr(senses, change)
    pr = partial_spearman(senses, change, logf)
    terts = np.quantile(logf, [1 / 3, 2 / 3])
    strata = {}
    for lbl, mask in [("low_freq", logf <= terts[0]),
                      ("mid_freq", (logf > terts[0]) & (logf <= terts[1])),
                      ("high_freq", logf > terts[1])]:
        r, pp = spearmanr(senses[mask], change[mask])
        strata[lbl] = {"rho": round(float(r), 4), "p": float(pp), "n": int(mask.sum())}
    print(f"\n=== {name} ===  (law predicts positive rho)")
    print(f"  n = {len(senses)}   rho = {rho:+.4f}   p = {p:.2e}   partial|freq = {pr:+.4f}")
    for lbl, d in strata.items():
        print(f"    {lbl:9}: rho = {d['rho']:+.4f}  p = {d['p']:.2e}  (n={d['n']})")
    return {"n": int(len(senses)), "spearman_rho": round(float(rho), 4), "p": float(p),
            "partial_rho_given_freq": round(pr, 4), "by_freq_tertile": strata}


def main():
    CACHE.mkdir(parents=True, exist_ok=True)
    targets = pick_targets()

    ctx_file = CACHE / "contexts.json.gz"
    if ctx_file.exists():
        contexts = json.loads(gzip.open(ctx_file, "rt", encoding="utf-8").read())
        print(f"loaded cached contexts ({ctx_file.name})")
    else:
        print("extracting raw contexts (stemming the raw eras once; slow-ish) ...")
        contexts = extract_contexts(targets)
        with gzip.open(ctx_file, "wt", encoding="utf-8") as f:
            f.write(json.dumps(contexts, ensure_ascii=False))
        print(f"cached -> {ctx_file.name}")
    for era in (ERA_A, ERA_B):
        n_ok = sum(1 for w in contexts[era] if len(contexts[era][w]) >= MIN_CTX)
        print(f"  {era}: {n_ok} targets with >= {MIN_CTX} contexts")

    vec_file = CACHE / "vectors.npz"
    if vec_file.exists():
        z = np.load(vec_file, allow_pickle=True)
        vecs = {era: dict(z[era].item()) for era in (ERA_A, ERA_B)}
        print("loaded cached vectors")
    else:
        vecs = encode(contexts)
        np.savez_compressed(vec_file, **{era: np.array(vecs[era], dtype=object)
                                         for era in vecs})
        print(f"cached -> {vec_file.name}")

    shared = sorted(set(vecs[ERA_A]) & set(vecs[ERA_B]))
    print(f"\nscorable words (>= {MIN_CTX} contexts in both eras): {len(shared)}")
    rng = random.Random(SEED)
    senses = np.array([targets[w]["senses"] for w in shared], dtype=float)
    logf = np.array([targets[w]["logf"] for w in shared])
    apds, withins, jsds = [], [], []
    for w in shared:
        A, B = vecs[ERA_A][w], vecs[ERA_B][w]
        apds.append(apd(A, B, rng))
        withins.append(0.5 * (apd(A, A, rng, exclude_diag=True)
                              + apd(B, B, rng, exclude_diag=True)))
        jsds.append(cluster_jsd(A, B))
    apds, withins, jsds = map(np.array, (apds, withins, jsds))
    corrected = apds - withins        # change proper: between-era minus within-era

    out = {"model": MODEL, "eras": [ERA_A, ERA_B], "n_ctx": N_CTX, "min_ctx": MIN_CTX,
           "scorable_words": len(shared), "tests": {}}
    out["tests"]["validity_senses_vs_within_era_spread"] = report(
        "Validity: WordNet senses vs WITHIN-era occurrence spread (synchronic)",
        senses, withins, logf)
    out["tests"]["senses_vs_raw_APD"] = report(
        "BanglaBERT Test 1a: WordNet senses vs raw between-era APD",
        senses, apds, logf)
    out["tests"]["senses_vs_corrected_APD"] = report(
        "BanglaBERT Test 1b: WordNet senses vs CORRECTED APD (between - within)",
        senses, corrected, logf)
    out["tests"]["senses_vs_clusterJSD"] = report(
        f"BanglaBERT Test 2: WordNet senses vs usage-cluster JSD (k={KMEANS_K})",
        senses, jsds, logf)

    # convergent validity: does corrected APD agree with the static ChangeScore?
    import csv
    cs = {nfc(r["word"]): float(r["change_score"])
          for r in csv.DictReader(open(RESULTS / "laws_data.csv", encoding="utf-8-sig"))}
    mask = np.array([w in cs for w in shared])
    rho_c, p_c = spearmanr(corrected[mask], np.array([cs[w] for w in shared if w in cs]))
    print(f"\nconvergent validity: corrected APD vs static ChangeScore "
          f"rho = {rho_c:+.4f}  p = {p_c:.2e}")
    out["convergent_validity_correctedAPD_vs_ChangeScore"] = {
        "spearman_rho": round(float(rho_c), 4), "p": float(p_c)}
    (RESULTS / "law2_bert_summary.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwrote law2_bert_summary.json -> {RESULTS}")


if __name__ == "__main__":
    main()
