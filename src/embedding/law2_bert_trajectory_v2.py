"""Trajectory extension of the BanglaBERT Law-of-Polysemy test: all five eras.

The endpoint test (law2_bert_v2.py) compares 1950-1970 with 2010-2025 directly, so a
word whose sense mix drifts and partly returns (meanders) nets out to a small score.
The Law of Innovation concerns the RATE of change, and polysemous words are exactly
the ones with room to meander. Here change is measured as the cumulative
within-era-corrected APD path across consecutive eras:

    path(w)    = sum over consecutive era pairs of  APD(e_i, e_i+1) - within-baseline
    meander(w) = path(w) - corrected endpoint distance   (wandering beyond net drift)

Tests (law predicts positive rho, frequency-partialled):
    senses vs path over the four main eras (primary; matches the paper's span)
    senses vs path over all five eras (with the pre-1950 anchor)
    senses vs meander

Bonus output: per-era-pair corrected APD trajectories for the case-study words
(ডিজিটাল, দারুণ, স্বাধীনতা), usable for an era-resolved change figure.

Reuses law2_bert_v2's target sample, extraction, encoder, and caches; the two eras
already encoded are loaded from cache, only pre_1950 / 1970_1990 / 1990_2010 are new.

Run:  .venv/bin/python src/embedding/law2_bert_trajectory_v2.py
Output: results_v2/law2_bert_trajectory_summary.json
"""
import gzip
import json
import random
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "embedding"))
import law2_bert_v2 as base  # noqa: E402  (helpers: stem, apd, encode, report, ...)

RESULTS = ROOT / "results_v2"
CACHE = RESULTS / "bert_cache"
RAW = ROOT / "corpus_build" / "epub_dataset"
ERAS = ["pre_1950", "1950_1970", "1970_1990", "1990_2010", "2010_2025"]
MAIN = ERAS[1:]                       # the paper's four main eras
CASE_WORDS = ["ডিজিটাল", "দারুণ", "স্বাধীনতা"]   # bonus trajectories (not in the law test)
N_CTX, MIN_CTX, SEED = base.N_CTX, base.MIN_CTX, base.SEED


def extract_era(era, targets):
    """Reservoir-sample up to N_CTX raw contexts per target for one era."""
    out = {w: [] for w in targets}
    seen = {w: 0 for w in targets}
    rng = random.Random(SEED)
    files = sorted((RAW / era).glob("*.txt"))
    print(f"  scanning {era}: {len(files)} files")
    for fp in files:
        try:
            text = base.nfc(fp.read_text(encoding="utf-8", errors="ignore"))
        except Exception:
            continue
        for sent in base.SENT_SPLIT.split(text):
            raw_toks = sent.split()
            if not (3 <= len(raw_toks) <= 400):
                continue
            for i, rt in enumerate(raw_toks):
                tok = base.EDGE_PUNCT.sub("", rt)
                if not tok or base.HAS_ASCII.search(tok):
                    continue
                st = base.stem(tok)
                if st not in out:
                    continue
                seen[st] += 1
                lo = max(0, i - base.HALF_WIN)
                rec = (" ".join(raw_toks[lo:i + base.HALF_WIN + 1]), i - lo, tok)
                if len(out[st]) < N_CTX:
                    out[st].append(rec)
                else:
                    j = rng.randrange(seen[st])
                    if j < N_CTX:
                        out[st][j] = rec
    return out


def era_vectors(era, targets):
    """Vectors for one era: from the endpoint run's cache when available,
    otherwise extract + encode + cache."""
    old = CACHE / "vectors.npz"
    if era in ("1950_1970", "2010_2025") and old.exists():
        z = np.load(old, allow_pickle=True)
        return dict(z[era].item())
    vf = CACHE / f"vectors_{era}.npz"
    if vf.exists():
        z = np.load(vf, allow_pickle=True)
        return dict(z["v"].item())
    cf = CACHE / f"contexts_{era}.json.gz"
    if cf.exists():
        ctx = json.loads(gzip.open(cf, "rt", encoding="utf-8").read())
    else:
        ctx = extract_era(era, targets)
        with gzip.open(cf, "wt", encoding="utf-8") as f:
            f.write(json.dumps(ctx, ensure_ascii=False))
    vecs = base.encode({era: ctx})[era]
    np.savez_compressed(vf, v=np.array(vecs, dtype=object))
    return vecs


def corrected(A, B, rng, wcache, ka, kb):
    if ka not in wcache:
        wcache[ka] = base.apd(A, A, rng, exclude_diag=True)
    if kb not in wcache:
        wcache[kb] = base.apd(B, B, rng, exclude_diag=True)
    return base.apd(A, B, rng) - 0.5 * (wcache[ka] + wcache[kb])


def main():
    CACHE.mkdir(parents=True, exist_ok=True)
    targets = base.pick_targets()
    extras = {w: {"senses": None, "logf": None} for w in CASE_WORDS if w not in targets}
    all_words = {**targets, **extras}
    print(f"targets: {len(targets)} law words + {len(extras)} case-study extras")

    vecs = {}
    for era in ERAS:
        vecs[era] = era_vectors(era, all_words)
        print(f"  {era}: {len(vecs[era])} words with >= {MIN_CTX} contexts")

    rng = random.Random(SEED)
    wcache = {}

    def seg(w, ea, eb):
        return corrected(vecs[ea][w], vecs[eb][w], rng, wcache, (w, ea), (w, eb))

    # ---- law tests --------------------------------------------------------------
    out = {"model": base.MODEL, "eras": ERAS, "n_ctx": N_CTX, "min_ctx": MIN_CTX,
           "tests": {}, "case_trajectories": {}}

    def battery(era_list, tag):
        shared = [w for w in targets
                  if all(w in vecs[e] for e in era_list)]
        senses = np.array([targets[w]["senses"] for w in shared], dtype=float)
        logf = np.array([targets[w]["logf"] for w in shared])
        pairs = list(zip(era_list[:-1], era_list[1:]))
        path = np.array([sum(seg(w, a, b) for a, b in pairs) for w in shared])
        endpoint = np.array([seg(w, era_list[0], era_list[-1]) for w in shared])
        meander = path - endpoint
        out["tests"][f"path_{tag}"] = base.report(
            f"senses vs corrected-APD PATH over {tag} ({len(era_list)} eras)",
            senses, path, logf)
        out["tests"][f"meander_{tag}"] = base.report(
            f"senses vs MEANDER (path minus endpoint) over {tag}",
            senses, meander, logf)
        return shared

    battery(MAIN, "main4")
    battery(ERAS, "all5")

    # ---- case-study trajectories --------------------------------------------------
    for w in CASE_WORDS:
        traj = {}
        for a, b in zip(ERAS[:-1], ERAS[1:]):
            if w in vecs.get(a, {}) and w in vecs.get(b, {}):
                traj[f"{a}->{b}"] = round(seg(w, a, b), 4)
        out["case_trajectories"][w] = traj
        print(f"  trajectory {w}: {traj}")

    (RESULTS / "law2_bert_trajectory_summary.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwrote law2_bert_trajectory_summary.json -> {RESULTS}")


if __name__ == "__main__":
    main()
