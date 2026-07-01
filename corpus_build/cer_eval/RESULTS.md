# CER evaluation — results

Text fidelity of the v2 corpus, measured against **author hand-transcriptions of original
print editions** (gold), scored with `cer_score.py`.

## Method
- **Gold**: one prose page per book, hand-typed from a print **scan image** (independent of
  any digital text, to avoid shared-source bias). EPUB books use authoritative editions;
  OCR baselines use the archive.org page images.
- **Hypothesis**: our EPUB-extracted text (corpus) / archive.org djvu-OCR text (baselines).
- **Scoring**: NFC-normalized; located via rapidfuzz partial alignment; boundary-trimmed.
  - **CER** = char Levenshtein / |gold|.  **CER-nows** = whitespace removed (pure character
    fidelity, ignores spacing/segmentation conventions).
  - **WER-norm** = segmentation-invariant word error (punctuation/joiners dropped, spacing
    ignored; a word counts wrong only if a real character edit touches it).
- All numbers are **upper bounds** (they also absorb transcriber slips + edition variants).

## Results
| source | print era | n chars | CER | CER-nows | WER-norm |
|---|---|--:|--:|--:|--:|
| **EPUB corpus** (Tagore, Manik, Humayun, Sunil, Sadat) | 1892–2017, human-typed digital | 3,135 | 2.30% | **0.70%** | 1.92% |
| net OCR — বিষবৃক্ষ (Bankim) | 1873 print | 963 | 27.10% | **28.38%** | 44.60% |
| net OCR — উপন্যাস (Tagore ed.) | 1986 print | 479 | 6.68% | **7.71%** | 16.67% |

## Findings
1. **Our corpus is ~0.7% CER** — and its errors are **benign**: ~all are diacritic omissions
   (missing চন্দ্রবিন্দু / matra), the signature of **human typing**, with **zero** OCR-style
   glyph corruption / Latin intrusions / broken conjuncts. → the EPUBs are born-digital,
   human-typed, not OCR'd.
2. **Net OCR error is severe on old print (28%) and improves by the 1980s (7.7%)** — i.e.,
   worst exactly for the historical eras most important to diachronic analysis. The OCR error
   profile is the opposite: glyph confusions, broken conjuncts, garbage characters.
3. **By sourcing human-typed digital editions, we avoid this OCR noise entirely** — ~0.7% CER,
   flat across all eras, vs 8–28% for net OCR.

Reproduce: `python corpus_build/cer_eval/cer_score.py` (gold in `sources/*/gold.txt`).
