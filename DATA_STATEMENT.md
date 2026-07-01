# Data statement

## Sources
The corpus is drawn from two kinds of text:

- **Literature (all eras):** human-typed digital book editions (EPUB), shared by a
  community of Bangla readers. These are typed transcriptions of published editions,
  not page scans, which is why the text is clean (measured ~0.7% character error rate;
  see `corpus_build/cer_eval/RESULTS.md`).
- **Modern news (2010–2025):** digital newspaper text, chiefly the online archive of
  *Prothom Alo*.

Each work is dated by its **first-publication year** (not the edition on hand).
Translations, religious scripture, and reference works (dictionaries, grammars) were
excluded. Duplicate copies of a work (standalone vs. inside a collected-works volume)
were deduplicated to a single copy.

## What `data_cleaned_v2/` actually contains
It is **processed, not verbatim** text. The cleaning pipeline
(`corpus_build/clean_corpus_v2.py`) applies, in order:

1. Unicode NFC normalisation.
2. Removal of digits, Latin-script runs, and non-Bangla punctuation.
3. A token filter (a token is kept only if it has ≥2 characters and ≥1 Bangla vowel
   or vowel sign).
4. Sentence segmentation at the Bangla full stop (`।`), rewritten as `<s>` markers.
5. **Stemming** of every word (a Bangla stemmer) and **stopword removal**.

Because stopwords, punctuation, digits, and formatting are discarded, and because
stemming is lossy and non-invertible, the original books **cannot be reconstructed
exactly** from these files. Word order within a sentence is preserved so that the
embedding context windows are meaningful.

## Copyright
Some source works are in the public domain (e.g. Rabindranath Tagore, Bankim Chandra
Chattopadhyay, Manik Bandyopadhyay); others are still under copyright. The processed
corpus is released here in good faith **for non-commercial research reproducibility**,
so that the results in the paper can be verified and extended. We do not claim
ownership of the underlying literary works. Anyone reusing the corpus is responsible
for their own compliance with applicable copyright law in their jurisdiction. If you
are a rights-holder and have a concern about a specific work, please contact the
authors and we will address it.

## CER evaluation data
`corpus_build/cer_eval/` contains the scorer and, per book, a single hand-transcribed
gold page (`gold.txt`) plus the archive OCR text used as a baseline. The full source
PDFs and page scans are **not** redistributed here; the file names and metadata in
each folder identify where they came from.
