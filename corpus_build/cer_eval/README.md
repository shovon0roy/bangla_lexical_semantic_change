# CER evaluation — materials

This folder measures the Character Error Rate (CER) of the corpus text against
hand-transcribed pages of the original print editions. Results are in `RESULTS.md`; the
scorer is `cer_score.py`.

## What is in each book folder (`sources/<book>/`)
- `gold.txt` — the ground truth: one page **hand-typed from a print-scan image** of that
  book (lines beginning with `#` are comments, ignored by the scorer).
- the hypothesis that gold is scored against:
  - EPUB books (our corpus): `extracted.txt` — the aligned passage of our corpus text for
    the same page.
  - baseline books: `ocr_text.txt` — the archive.org djvu-OCR text.

| folder | author / work | print era | role |
|---|---|---|---|
| `1_tagore_kabuliwala_1892_f649` | Tagore — কাবুলিওয়ালা | 1892 | our corpus |
| `2_manik_padmanadirmajhi_1936_f1001` | Manik — পদ্মা নদীর মাঝি | 1936 | our corpus |
| `3_humayun_dure-kothao_1988_f1780` | Humayun Ahmed — দূরে কোথাও | 1988 | our corpus |
| `4_sunil_ranu-o-bhanu_1998_f1589` | Sunil — রানু ও ভানু | 1998 | our corpus |
| `5_sadat_manabjanam_2017_f2172` | Sadat Hossain — মানবজনম | 2017 | our corpus |
| `baseline_old_bishabriksha_bankim_1873_IA` | Bankim — বিষবৃক্ষ | 1873 | archive OCR baseline |
| `baseline_1980s_tagore-upanyas-print_1986_IA` | Tagore novels (1986 ed.) | 1986 | archive OCR baseline |

The original scan PDFs / page images are **not redistributed** (copyright, size); the
`gold.txt` + `extracted.txt` / `ocr_text.txt` pairs are enough to reproduce the CER.

## Reproduce
```bash
python corpus_build/cer_eval/cer_score.py
```
The scorer NFC-normalises both sides, locates the gold passage inside the hypothesis
(rapidfuzz partial alignment), and reports CER, CER-nows (whitespace removed), WER and
WER-norm, then aggregates EPUB corpus vs OCR baseline.

## The rule that makes the measurement valid
Each `gold.txt` was **hand-typed from a scanned page image** — typing what the eye sees in
the print. We never copy-pasted from a digital text: a digital copy could share our EPUB's
origin and match trivially (~0%), proving nothing.
