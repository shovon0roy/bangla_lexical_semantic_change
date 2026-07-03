# Mapping the Semantic Frontier of Bangla (1950–2025)

Code, trained models, derived data, and the diachronic corpus for the paper
**"Mapping the Semantic Frontier of Bangla (1950–2025): A Hybrid N-gram and
Procrustes-Aligned Embedding Analysis of Diachronic Lexical Change"**
by Sagor Chanda, Shovon Roy, and Muhammad Masroor Ali
(Bangladesh University of Engineering and Technology).

This is the first computational study of how Bangla word meanings have shifted across
its recent history. We build a large, time-stratified corpus (≈24M cleaned tokens,
1950–2025 plus a pre-1950 anchor) from human-typed digital editions rather than OCR,
and run a hybrid pipeline: an n-gram module (frequency + PPMI collocates) and
an embedding module (per-era Word2Vec aligned by Orthogonal Procrustes, fused into a
single ChangeScore). We test the two "universal laws" of semantic change on Bangla.

## What is in this release

```
src/                 Analysis pipeline (Python). *_v2.py are the scripts used for the paper.
corpus_build/        Corpus cleaning (clean_corpus_v2.py), the name-filter seed, and the
                     CER evaluation (scorer + one-page gold transcriptions).
data_cleaned_v2/     The cleaned corpus (stemmed, stopword-removed). NOT in this repo —
                     download it from the Zenodo dataset record (see "Data" below).
models_v2/           Per-era Word2Vec embeddings. NOT included — regenerated from the
                     corpus by src/embedding/train_embeddings_v2.py (deterministic).
results_v2/          Every derived table and figure input: drift/ChangeScore CSVs, laws,
                     collocates, Spiker ranking, case studies, summaries.
book/figures/        The figures used in the paper.
REPRODUCE.md         Stage-by-stage run order for the whole pipeline.
requirements-lock.txt  Pinned dependency versions.
DATA_STATEMENT.md    Corpus provenance, processing, and the copyright statement.
CITATION.cff         How to cite this release.
LICENSE              Licensing (code vs. data).
```

## Reproducing the results

See **REPRODUCE.md** for the full run order. In brief:

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements-lock.txt

# 1. Download the five era files (+ stats.json) from the Zenodo dataset record
#    (DOI below) into data_cleaned_v2/.
# 2. Train the per-era embeddings from the corpus (deterministic):
PYTHONHASHSEED=0 python src/embedding/train_embeddings_v2.py
# 3. Run the analysis and regenerate the figures:
PYTHONHASHSEED=0 python src/embedding/analyze_drift_v2.py
PYTHONHASHSEED=0 python src/embedding/laws_v2.py
PYTHONHASHSEED=0 python src/embedding/spikers_v2.py
python src/N-gram/collocates_v2.py
PYTHONHASHSEED=0 python src/make_figures_v2.py
```

An optional battery of Law-of-Polysemy tests (IndoWordNet dictionary senses, an
embedding-free collocate-JSD test, and BanglaBERT occurrence-level tests with
within-era baseline corrections) lives in `src/embedding/law2_wordnet_v2.py` and
`law2_bert_*.py`; see stage 6c of `REPRODUCE.md` and the optional-extras block in
`requirements-lock.txt`. The sampled contexts these tests need are shipped in
`results_v2/bert_cache/` (the raw un-stemmed editions themselves are not
redistributed).

Determinism: scripts re-exec with `PYTHONHASHSEED=0`, Word2Vec is trained with
`workers=1` and a fixed seed, and numpy/random are seeded, so retraining on the same
corpus reproduces the same models, tables, and figures.

## Data

The cleaned corpus is **not stored in this repository**. Download the five era files
(`pre_1950.txt`, `1950_1970.txt`, `1970_1990.txt`, `1990_2010.txt`, `2010_2025.txt`,
plus `stats.json`) from the Zenodo dataset record into `data_cleaned_v2/`:

> **Corpus DOI:** [10.5281/zenodo.21100721](https://doi.org/10.5281/zenodo.21100721)

The corpus is derived from published Bangla literary editions and modern newspaper text
and is released for research reproducibility. It is processed, not verbatim: words are
stemmed, stopwords are removed, and punctuation, digits, and formatting are stripped, so
the original books cannot be reconstructed exactly. Full provenance and the copyright
statement are in **DATA_STATEMENT.md**. Code is MIT licensed; see **LICENSE**.

## How this is distributed

Two pieces, cross-linked:

- **This GitHub repository** — the code, derived result tables (`results_v2/`), figures,
  metadata, and documentation.
- **A Zenodo dataset record (its own DOI)** — the cleaned corpus, as the five era files.
  The corpus is not kept in Git; download it from the DOI into `data_cleaned_v2/`.

Trained models are not distributed at all — they are regenerated from the corpus by the
code (`src/embedding/train_embeddings_v2.py`), deterministically. `.gitignore` keeps the
extracted corpus (`data_cleaned_v2/`) and any local `models_v2/` out of the repository.

## Citing

If you use this work, please cite the paper (see `CITATION.cff`), and as appropriate:
- **Code** (this repository, v1.1.0): DOI [10.5281/zenodo.21141394](https://doi.org/10.5281/zenodo.21141394)
- **Corpus** (dataset): DOI [10.5281/zenodo.21100721](https://doi.org/10.5281/zenodo.21100721)

## Contact

Shovon Roy — shovon0roy@gmail.com · Sagor Chanda — sagorchanda.official@gmail.com
