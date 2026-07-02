# Reproducing the corpus & analysis

This document is the canonical run-order for regenerating the diachronic Bangla corpus
and the semantic-change results from source. It is meant to accompany the published code
(DOI artifact for the journal paper). Each stage is a single command with explicit
inputs/outputs; determinism notes are called out where results depend on seeds/versions.

## 0. Environment
```bash
python3 -m venv .venv                       # Python 3.12.3
.venv/bin/pip install -r requirements-lock.txt
```
Pinned versions live in `requirements-lock.txt`. Stemming output is
**version-dependent** (`bangla-stemmer==1.0`), so the pin is part of reproducibility.

All commands below are run from the repo root with `.venv/bin/python` (absolute path is
safest — the scripts resolve their own paths, but the venv is at the repo root).

## Pipeline overview
```
Zenodo DOI ──download──▶ data_cleaned_v2/ ──train──▶ models_v2/ ──▶ ChangeScore / laws / n-gram / figures
```

## 1. Get the cleaned corpus → `data_cleaned_v2/`
The cleaned corpus is distributed as a Zenodo dataset,
**DOI [10.5281/zenodo.21100721](https://doi.org/10.5281/zenodo.21100721)**. Download the
five era files (`pre_1950.txt`, `1950_1970.txt`, `1970_1990.txt`, `1990_2010.txt`,
`2010_2025.txt`) plus `stats.json` into `data_cleaned_v2/`.
→ 5 era files, **23.8M clean tokens**.

The corpus was produced from digital book editions plus Prothom Alo news by
`corpus_build/clean_corpus_v2.py` (NFC → English/digit strip → punctuation strip →
sentence split → stem → stopword/invalid-token filter → ` <s>\n` format), with a
per-author per-era cap so no single author dominates. That script is included as the
authoritative record of the cleaning. The raw EPUBs and the hand-curated dating list are
not redistributed, so building the corpus from source is documentation rather than a
runnable stage here. Stemming is version-dependent (`bangla-stemmer==1.0`), which is why
the version is pinned.

## 4. Train per-era embeddings → `models_v2/`  *(done)*
```bash
.venv/bin/python src/embedding/train_embeddings_v2.py    # --epochs 5 --min-count 5 --seed 42
```
Deterministic by construction: the script re-execs with `PYTHONHASHSEED=0`, trains with
`workers=1` + fixed `seed`, seeds numpy/random. Skip-gram, dim 300, window 5, negative 10,
min_count 5, 5 epochs. → 5 models `word2vec_<era>.model` + `train_meta.json` (records
gensim/numpy/python versions, params, seed, per-era vocab). Vocab 42K–63K/era; ~20 min total.

## 5. Align + ChangeScore → `results_v2/`  *(done)*
```bash
.venv/bin/python src/embedding/analyze_drift_v2.py        # Procrustes + ChangeScore, all era pairs
.venv/bin/python src/embedding/filter_propernouns_v2.py --thresh 0.3 --rounds 8 --knn 20   # name-filtered candidate lists
```
ChangeScore = Δcos/2 + (1 − Jaccard@50). Outputs `drift_<A>_<B>.csv`, `top_changed_*`,
`top_stable_*`, `summary.json`. The proper-noun filter (seed `corpus_build/bn_name_seed.txt`
+ embedding-neighbour propagation, params thresh 0.3 / rounds 8 / knn 20) writes
`top_changed_clean_*` — a heuristic candidate cleaner. The seed was extended with the
modern-era character/person names that leaked into the frequency spikers (e.g. ছোটাচ্চু,
মিতিন, লাবনী) so the filtered lists are reproducible; literary corpora still need light
manual curation at the very top.

## 6. Laws → `results_v2/`  *(done)*
```bash
.venv/bin/python src/embedding/laws_v2.py
```
Law 1 (frequency↔change, Spearman) and Law 2 (polysemy proxy = neighbour dispersion ↔
change). Writes `laws_summary.json`, `laws_data.csv`, and the two scatter PNGs.

```bash
.venv/bin/python src/embedding/laws_clean_v2.py   # de-confound: name-excluded + freq-stratified
```
Sensitivity analysis → `laws_clean_summary.json`. Finding: name exclusion barely moves the
laws (NOT a proper-noun artifact); raising the frequency floor (reliable vocab) *strengthens*
both — Law 1 (conformity) confirmed; Law 2 a robust significant NEGATIVE rho (→ −0.51),
differing from the original paper's null.

```bash
.venv/bin/python src/embedding/law2_polysemy_v2.py     # dispersion vs true sense-count
```
→ `law2_polysemy_summary.json`. Resolves Law 2: the negative result is GENERICNESS
(dispersion), not polysemy. A proper sense-cluster count (connected components of the
neighbour graph) is only weakly and *negatively* correlated with change (ρ ≈ −0.06,
p < 0.001 at working edge thresholds 0.55–0.65; the ρ≈0 seen at edge 0.45 is a degenerate
all-in-one-cluster artefact) — never the positive the law predicts → Law of Innovation
does NOT hold for Bangla. The paper reports ρ = −0.066 (sense-count, edge 0.60, full shared vocabulary — the figure and the summary JSON now agree).

## 6c. Law-2 artifact-free battery (dictionary senses + contextual embeddings)
```bash
.venv/bin/python src/embedding/law2_wordnet_v2.py          # Tests A/B + dispersion diagnostic
.venv/bin/python src/embedding/law2_bert_v2.py             # BanglaBERT endpoint tests (raw/corrected APD)
.venv/bin/python src/embedding/law2_bert_trajectory_v2.py  # all-five-era path/meander tests
.venv/bin/python src/embedding/law2_bert_wsd_v2.py         # dictionary-sense WSD rebalancing test
```
Replaces the embedding-derived polysemy proxy with **IndoWordNet (Bengali) sense counts**
(`pyiwn`; 4,329 of the 10,517 laws-vocabulary words match), and the change measure with
(i) the ChangeScore, (ii) an embedding-free collocate JSD, and (iii) BanglaBERT
occurrence-level APD with a **within-era baseline correction** (raw APD without that
correction measures synchronic polysemy, not change). Findings
(`law2_wordnet_summary.json`, `law2_bert_summary.json`): every fair test is null — the
Law of Innovation does not hold on this corpus under any instrument; the earlier negative
signs were proxy artifacts; dispersion shares only rho=0.15 with dictionary senses;
corrected APD agrees with the static ChangeScore at rho=0.34 (convergent validity).
NOTE: the BERT scripts sample raw sentences from `corpus_build/epub_dataset/` (the
un-stemmed editions, not redistributed). The sampled context snippets ARE shipped in
`results_v2/bert_cache/contexts*.json.gz`, so the encoding + analysis stages reproduce
without the raw corpus; extraction itself requires it. Optional deps: see the extras
block in `requirements-lock.txt`.

## 7. Case studies → `results_v2/`  *(done)*
```bash
.venv/bin/python src/embedding/case_studies_v2.py
```
Neighbour shift (early vs late era) + PCA drift figures for showcase words. The three
case studies used in the paper: ডিজিটাল (emergence: gadgets→AI/fintech), দারুণ (reversal /
amelioration: dire→wonderful, negative-affect→sport collocates), স্বাধীনতা (anchoring:
independence→1971 commemoration). Others explored but not used: গাজা (cannabis→Palestine;
too sparse pre-2010), গুম (a sound→enforced disappearance), সাবেক. কুল was dropped — the
literary corpus lacks the "cool" slang shift and stemming conflates its senses.
→ `case_studies.md` + `case_*.png`.

## 8. Frequency / n-gram prong → `results_v2/`  *(done)*
```bash
.venv/bin/python src/N-gram/freq_trends_v2.py
```
Per-era per-million frequencies → emergent (ট্রাম্প, বিশ্বকাপ, উপজেলা) and declining words.
Headline: top decliners are সাধু-ভাষা archaic verbs (বলিল/কহিল/করিল/হইল) — recovers the
known Sadhu→Cholito shift unsupervised. → `emergent_words.csv`, `declining_words.csv`,
`freq_trajectories.csv`, `figure_frequency_trends.png`.

## 9. Paper figures & tables → `book/figures/`, `results_v2/`  *(done)*
```bash
.venv/bin/python src/N-gram/collocates_v2.py              # PMI collocates: দারুণ, স্বাধীনতা, ডিজিটাল
PYTHONHASHSEED=0 .venv/bin/python src/embedding/spikers_v2.py   # name-filtered Spiker ranking → spikers_clean.csv
PYTHONHASHSEED=0 .venv/bin/python src/make_figures_v2.py  # all 6 paper data-figures into book/figures/
```
`collocates_v2.py` → `collocates_<word>.json` (PPMI: window 5, min co-occur 3, clamped at 0; counts stem variants such as স্বাধীনত with স্বাধীনতা so genitive contexts are not lost), feeds the heatmaps.
`spikers_v2.py` → `spikers_clean.csv`: words nearly absent pre-1990 (early pm < 1) ranked by
modern pm, with seed+propagation name filter applied → paper's Spikers table (ট্রাম্প, উপজেলা,
বিএনপি, বিশ্বকাপ, জামায়াত, …). `make_figures_v2.py` re-execs with `PYTHONHASHSEED=0`, uses a
dual-script font (`DejaVu Sans` + `Kalpurush`), writes the 6 data figures the manuscript
`\includegraphics` (same filenames): `changescore_distribution`, `frequency_trends`,
`drift_digital` (UMAP), `heatmap_darun`, `heatmap_swadhinata`, `law_frequency` + `law_polysemy`.
The দারুণ and স্বাধীনতা heatmaps use curated, interpretable collocate sets (dire→wonderful;
anticolonial→commemoration→rights); the law figure uses the full shared vocabulary. Then compile `journal_paper/paper.tex` with XeLaTeX (2 passes).

---
### Determinism checklist
- [x] Dependencies pinned (`requirements-lock.txt`)
- [x] Corpus cleaning is deterministic given the pinned stemmer (`bangla-stemmer==1.0`)
- [x] Word2Vec seed + `workers=1` + `PYTHONHASHSEED=0` (stage 4 — `models_v2/train_meta.json`)
- [ ] Record corpus token counts + model hyperparameters with results (stage 5)
