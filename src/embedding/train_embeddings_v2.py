"""Train one Word2Vec model per era on the v2 cleaned corpus — DETERMINISTICALLY.

Reproducibility (the models must be bit-identical on re-run, for the DOI artifact):
  * re-exec with PYTHONHASHSEED=0 (fixes string hashing / vocab tie order)
  * workers=1            (no async/HOGWILD races -> deterministic SGD)
  * fixed seed           (init + negative sampling)
  * numpy/random seeded
Hyperparameters match the original paper's analysis (skip-gram, dim 300, window 5,
negative 10); min_count is raised to 5 for the ~20x larger corpus. Every choice is
recorded in models_v2/train_meta.json next to the models.

Input : data_cleaned_v2/<era>.txt   (' <s>\n' sentence format)
Output: models_v2/word2vec_<era>.model   (+ train_meta.json)

Run:  .venv/bin/python src/embedding/train_embeddings_v2.py
      (optional) --epochs 5 --min-count 5 --seed 42
"""
import os
import sys

# --- guarantee deterministic hashing BEFORE anything else imports ------------
if os.environ.get("PYTHONHASHSEED") != "0":
    os.environ["PYTHONHASHSEED"] = "0"
    os.execv(sys.executable, [sys.executable] + sys.argv)

import argparse  # noqa: E402
import json  # noqa: E402
import random  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import gensim  # noqa: E402
from gensim.models import Word2Vec  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data_cleaned_v2"
OUT = ROOT / "models_v2"
ERAS = ["pre_1950", "1950_1970", "1970_1990", "1990_2010", "2010_2025"]

# core hyperparameters (skip-gram + negative sampling, as in the original paper)
PARAMS = dict(vector_size=300, window=5, sg=1, negative=10, min_count=5, workers=1)


def read_sentences(path):
    """Each ' <s>\n'-delimited unit -> a list of tokens."""
    text = path.read_text(encoding="utf-8")
    out = []
    for chunk in text.replace("<s/>", "<s>").split("<s>"):
        toks = chunk.split()
        if toks:
            out.append(toks)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--min-count", type=int, default=PARAMS["min_count"])
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    OUT.mkdir(parents=True, exist_ok=True)
    params = {**PARAMS, "min_count": args.min_count, "epochs": args.epochs, "seed": args.seed}

    meta = {"gensim": gensim.__version__, "numpy": np.__version__,
            "python": sys.version.split()[0], "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
            "params": params, "eras": {}}
    print(f"deterministic train | gensim {gensim.__version__} | seed {args.seed} | "
          f"workers=1 | PYTHONHASHSEED={os.environ.get('PYTHONHASHSEED')}\n")

    for era in ERAS:
        fp = DATA / f"{era}.txt"
        if not fp.exists():
            print(f"  [skip] {era}: no file")
            continue
        sents = read_sentences(fp)
        ntok = sum(len(s) for s in sents)
        t0 = time.time()
        model = Word2Vec(sentences=sents, seed=args.seed, epochs=args.epochs,
                         vector_size=params["vector_size"], window=params["window"],
                         sg=params["sg"], negative=params["negative"],
                         min_count=params["min_count"], workers=1)
        dt = time.time() - t0
        outp = OUT / f"word2vec_{era}.model"
        model.save(str(outp))
        meta["eras"][era] = {"sentences": len(sents), "tokens": ntok,
                             "vocab": len(model.wv), "train_sec": round(dt, 1)}
        print(f"  {era:10} sents={len(sents):>7,} tok={ntok:>10,} "
              f"vocab={len(model.wv):>7,}  ({dt:.0f}s) -> {outp.name}")

    (OUT / "train_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2),
                                         encoding="utf-8")
    print(f"\nwrote {len(meta['eras'])} models + train_meta.json -> {OUT}")


if __name__ == "__main__":
    main()
