"""The most literal test of the Law of Polysemy: dictionary-sense rebalancing.

Assigns every BanglaBERT occurrence vector (cached by law2_bert_v2.py for 1950-1970
and 2010-2025) to its nearest IndoWordNet sense, where each sense is represented by
the BanglaBERT encoding of its Bengali gloss plus first example (Lesk in embedding
space). A word's change is then the shift of its distribution over ACTUAL dictionary
senses between the two eras.

Two safeguards keep the test honest:
  * polysemous words only (k >= 2): a monosemous word has JSD = 0 by definition, so
    including k = 1 would manufacture a positive correlation mechanically;
  * the between-era JSD is normalised by log2(k) (the JSD ceiling grows with k) and
    corrected by a within-era split-half baseline (sampling noise alone produces
    nonzero JSD, more so for words with more senses).

The law predicts: among polysemous words, those with more senses rebalance more
(positive rho of senses vs corrected sense-JSD, frequency-partialled).

Run:  .venv/bin/python src/embedding/law2_bert_wsd_v2.py
Output: results_v2/law2_bert_wsd_summary.json
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "embedding"))
import law2_bert_v2 as base  # noqa: E402

RESULTS = ROOT / "results_v2"
CACHE = RESULTS / "bert_cache"
ERA_A, ERA_B = "1950_1970", "2010_2025"
MAX_SENSES = 12          # cap pathological entries
SEED = 0


def sense_texts():
    """word -> list of 'gloss + first example' strings, one per IndoWordNet synset."""
    import pyiwn
    iwn = pyiwn.IndoWordNet(lang=pyiwn.Language.BENGALI)
    texts = {}
    for w in iwn.all_words():
        if "_" in w:
            continue
        try:
            ss = iwn.synsets(w)
        except Exception:
            continue
        entries = []
        for s in ss[:MAX_SENSES]:
            try:
                g = s.gloss() or ""
                ex = (s.examples() or [""])[0] or ""
                t = base.nfc((g + " " + ex).strip())
                if t:
                    entries.append(t)
            except Exception:
                pass
        if len(entries) >= 2:                 # polysemous only
            texts[base.nfc(w)] = entries
    return texts


def encode_texts(texts_flat):
    """Mean-pooled BanglaBERT vectors for a flat list of short texts."""
    import torch
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(base.MODEL)
    model = AutoModel.from_pretrained(base.MODEL)
    model.eval()
    torch.set_grad_enabled(False)
    out = []
    B = 32
    for bi in range(0, len(texts_flat), B):
        batch = texts_flat[bi:bi + B]
        enc = tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=96)
        h = model(**enc).last_hidden_state                      # [B,T,H]
        mask = enc["attention_mask"].unsqueeze(-1).float()
        v = (h * mask).sum(1) / mask.sum(1)                     # mean over real tokens
        out.append(v.numpy())
        if (bi // B) % 20 == 0:
            print(f"  gloss encoding {bi}/{len(texts_flat)}", file=sys.stderr)
    return np.vstack(out)


def jsd_counts(ca, cb):
    p = ca / ca.sum()
    q = cb / cb.sum()
    m = 0.5 * (p + q)
    def kl(a, b):
        mask = a > 0
        return float(np.sum(a[mask] * np.log2(a[mask] / b[mask])))
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def main():
    # occurrence vectors from the endpoint run's cache
    z = np.load(CACHE / "vectors.npz", allow_pickle=True)
    vecs = {era: dict(z[era].item()) for era in (ERA_A, ERA_B)}
    shared = sorted(set(vecs[ERA_A]) & set(vecs[ERA_B]))

    targets = base.pick_targets()
    stexts = sense_texts()
    words = [w for w in shared if w in targets and w in stexts
             and targets[w]["senses"] and targets[w]["senses"] >= 2]
    print(f"polysemous scorable words with usable glosses: {len(words)}")

    # encode all sense glosses in one go
    flat, index = [], {}
    for w in words:
        index[w] = (len(flat), len(stexts[w]))
        flat.extend(stexts[w])
    print(f"encoding {len(flat)} sense-gloss texts ...")
    gv = encode_texts(flat)
    gv = gv / np.linalg.norm(gv, axis=1, keepdims=True)

    senses_arr, logf, raw, corr, ent = [], [], [], [], []
    flips = 0
    flip_words = []
    for w in words:
        off, k = index[w]
        G = gv[off:off + k]                                    # [k, H]
        dists = {}
        halves = {}
        for era in (ERA_A, ERA_B):
            V = vecs[era][w]
            V = V / np.linalg.norm(V, axis=1, keepdims=True)
            lab = (V @ G.T).argmax(axis=1)                     # nearest sense
            dists[era] = np.bincount(lab, minlength=k).astype(float)
            h1 = np.bincount(lab[0::2], minlength=k).astype(float)
            h2 = np.bincount(lab[1::2], minlength=k).astype(float)
            halves[era] = jsd_counts(h1, h2) / np.log2(k)
        between = jsd_counts(dists[ERA_A], dists[ERA_B]) / np.log2(k)
        baseline = 0.5 * (halves[ERA_A] + halves[ERA_B])
        senses_arr.append(k)
        logf.append(targets[w]["logf"])
        raw.append(between)
        corr.append(between - baseline)
        p = dists[ERA_A] / dists[ERA_A].sum()
        ent.append(float(-(p[p > 0] * np.log2(p[p > 0])).sum()) / np.log2(k))
        if dists[ERA_A].argmax() != dists[ERA_B].argmax():
            flips += 1
            flip_words.append({
                "word": w, "senses": k,
                "dominant_1950_1970": stexts[w][int(dists[ERA_A].argmax())][:90],
                "dominant_2010_2025": stexts[w][int(dists[ERA_B].argmax())][:90]})

    senses_arr = np.array(senses_arr, dtype=float)
    logf = np.array(logf)
    raw = np.array(raw)
    corr = np.array(corr)
    print(f"\nassignment diagnostics: mean normalised entropy = {np.mean(ent):.3f} "
          f"(0 = one sense used, 1 = uniform); majority-sense flips between eras = "
          f"{flips}/{len(words)}")

    out = {"model": base.MODEL, "eras": [ERA_A, ERA_B], "polysemous_words": len(words),
           "mean_assignment_entropy": round(float(np.mean(ent)), 4),
           "majority_sense_flips": flips, "flip_words": flip_words, "tests": {}}
    out["tests"]["senses_vs_raw_senseJSD"] = base.report(
        "WSD Test raw: senses vs normalised sense-JSD (uncorrected)",
        senses_arr, raw, logf)
    out["tests"]["senses_vs_corrected_senseJSD"] = base.report(
        "WSD Test corrected: senses vs sense-JSD minus split-half baseline",
        senses_arr, corr, logf)
    (RESULTS / "law2_bert_wsd_summary.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwrote law2_bert_wsd_summary.json -> {RESULTS}")


if __name__ == "__main__":
    main()
