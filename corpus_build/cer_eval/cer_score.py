"""Score Character Error Rate (CER) + Word Error Rate (WER) of our corpus text against
hand-transcribed gold pages.

For each book folder under sources/:
  * gold = the human transcription (sources/<book>/gold.txt, '#'-comment lines ignored)
  * our text = the EPUB-extracted text (epub works, by file id) or the djvu-OCR text
    (the two baseline_* books)
  * locate the gold passage inside our text (rapidfuzz partial alignment), take the
    matching span as the hypothesis, and compute
        CER = Levenshtein(hyp, gold) / len(gold)        (codepoint level, NFC)
        WER = Levenshtein(hyp_words, gold_words) / #gold_words
  * aggregate per source group: EPUB corpus vs OCR baseline.

Both sides are NFC-normalised and whitespace-collapsed (so layout/line-wrap differences
don't count as character errors — we measure text fidelity, not formatting).

Run (after gold.txt files are filled):
  .venv/bin/python corpus_build/cer_eval/cer_score.py
"""
import csv
import glob
import re
import sys
import unicodedata
from pathlib import Path

from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "corpus_build" / "cer_eval" / "sources"
EPUB = ROOT / "corpus_build" / "epub_dataset"
MANIFEST = EPUB / "manifest.csv"
# A baseline folder is identified by containing an `ocr_text.txt` (the archive.org djvu
# OCR we score against). Any other sources/* folder is an EPUB work (keyed by _f<id>).

nfc = lambda s: unicodedata.normalize("NFC", s or "")
WS = re.compile(r"\s+")


def norm(s):
    return WS.sub(" ", nfc(s)).strip()


def read_gold(folder):
    fp = folder / "gold.txt"
    if not fp.exists():
        return ""
    lines = [ln for ln in fp.read_text(encoding="utf-8").splitlines() if not ln.lstrip().startswith("#")]
    return norm(" ".join(lines))


_man = None


def our_text(folder_name):
    """The hypothesis text to score against gold, in priority order:
      1. sources/<folder>/ocr_text.txt   — archive.org djvu OCR (the baseline books)
      2. sources/<folder>/extracted.txt  — the aligned corpus passage saved beside gold
         (so the CER is reproducible here without the full epub_dataset)
      3. the full EPUB-extracted text, looked up by file id in epub_dataset/ (source repo)."""
    ocr = SRC / folder_name / "ocr_text.txt"
    if ocr.exists():
        return norm(ocr.read_text(encoding="utf-8"))
    ext = SRC / folder_name / "extracted.txt"
    if ext.exists():
        return norm(ext.read_text(encoding="utf-8"))
    m = re.search(r"_f(\d+)$", folder_name)
    if not m:
        return ""
    fid = m.group(1)
    global _man
    if _man is None:
        _man = list(csv.DictReader(open(MANIFEST, encoding="utf-8-sig")))
    parts = [(EPUB / r["path"]).read_text(encoding="utf-8")
             for r in _man if r["source_file"].split("_", 1)[0] == fid]
    return norm("\n".join(parts))


import difflib

# punctuation / joiners that segmentation conventions disagree on (not text content)
PUNCT = re.compile(r"[।॥|,.;:!?\-—–_/\\()\[\]{}'\"“”‘’…*‌‍]+")


def _content(s):
    return WS.sub(" ", PUNCT.sub(" ", nfc(s))).strip()


def wer_norm(gold, hyp):
    """Segmentation-invariant word error: drop punctuation/joiners, remove ALL spacing,
    align the character content, and mark a gold word wrong only if a real character edit
    (substitution/deletion) touches it. Pure spacing / merge / split differences don't count."""
    gwords = _content(gold).split()
    if not gwords:
        return 0.0
    gchars, cmap = [], []          # content chars of gold + their word index
    for wi, w in enumerate(gwords):
        for ch in w:
            gchars.append(ch); cmap.append(wi)
    hchars = [c for w in _content(hyp).split() for c in w]
    bad = set()
    for tag, i1, i2, _, _ in difflib.SequenceMatcher(None, gchars, hchars, autojunk=False).get_opcodes():
        if tag in ("replace", "delete"):
            bad.update(cmap[i1:i2])
    return len(bad) / len(gwords)


def _trim(hyp, gold):
    """Trim alignment-boundary junk: cut hyp to the span between its first and last
    block that actually matches gold (removes chapter numbers / markup grabbed at edges)."""
    import difflib
    blocks = [b for b in difflib.SequenceMatcher(None, hyp, gold, autojunk=False)
              .get_matching_blocks() if b.size >= 3]
    if not blocks:
        return hyp
    return hyp[blocks[0].a: blocks[-1].a + blocks[-1].size]


def score(gold, big):
    """Locate gold inside big; return (cer, cer_nospace, wer, hyp_span)."""
    al = fuzz.partial_ratio_alignment(gold, big)        # best matching substring of `big`
    hyp = _trim(big[al.dest_start:al.dest_end], gold)
    cer = Levenshtein.distance(hyp, gold) / max(len(gold), 1)
    # whitespace-removed: isolates character corruption from spacing/segmentation
    g_ns, h_ns = gold.replace(" ", ""), hyp.replace(" ", "")
    cer_ns = Levenshtein.distance(h_ns, g_ns) / max(len(g_ns), 1)
    gw, hw = gold.split(), hyp.split()
    wer = Levenshtein.distance(hw, gw) / max(len(gw), 1)
    wern = wer_norm(gold, hyp)
    return cer, cer_ns, wer, wern, hyp


def main():
    folders = sorted([d for d in SRC.iterdir() if d.is_dir()])
    rows = []
    print(f"{'book':40} {'chars':>7} {'CER':>7} {'CER-nows':>9} {'WER':>7} {'WER-norm':>9}")
    print("-" * 86)
    for d in folders:
        gold = read_gold(d)
        grp = "OCR" if (d / "ocr_text.txt").exists() else "EPUB"
        if len(gold) < 50:
            print(f"{d.name:40} {'(gold not transcribed yet)':>45}")
            continue
        big = our_text(d.name)
        if not big:
            print(f"{d.name:40} {'(our text not found!)':>45}")
            continue
        cer, cer_ns, wer, wern, hyp = score(gold, big)
        rows.append((d.name, grp, len(gold), cer, cer_ns, wer, wern))
        print(f"{d.name:40} {len(gold):>7,} {cer*100:>6.2f}% {cer_ns*100:>8.2f}% "
              f"{wer*100:>6.2f}% {wern*100:>8.2f}%")

    if not rows:
        print("\nNo gold transcriptions yet. Fill sources/<book>/gold.txt and re-run.")
        return

    print("\n=== aggregate by source (char-weighted) ===")
    for grp in ("EPUB", "OCR"):
        g = [r for r in rows if r[1] == grp]
        if not g:
            continue
        tot = sum(r[2] for r in g)
        agg = lambda i: sum(r[i] * r[2] for r in g) / tot
        print(f"  {grp:6}: {len(g)} books, {tot:,} chars -> CER {agg(3)*100:.2f}%  "
              f"CER-nows {agg(4)*100:.2f}%  WER {agg(5)*100:.2f}%  WER-norm {agg(6)*100:.2f}%")
    print("\nCER-nows = whitespace removed (pure character fidelity).")
    print("WER-norm = segmentation-invariant (punctuation/joiners dropped, spacing ignored,")
    print("a word counts wrong only if a real character edit touches it).")
    print("EPUB = our corpus; OCR = archive.org djvu baseline.")


if __name__ == "__main__":
    main()
