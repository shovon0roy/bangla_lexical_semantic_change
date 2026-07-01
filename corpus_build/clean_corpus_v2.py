"""Clean the epub dataset into a training-ready, per-era corpus.

The preprocessing pipeline that produced the released corpus:
  strip English/digits -> strip punctuation (keep । ? ! | as sentence delimiters)
  -> split into sentences -> STEM each word (bangla_stemmer) -> drop stopwords +
  invalid tokens (len 2-12, must contain a Bengali vowel, no all-same-char / all-digit)
  -> join sentences with ' <s>\n'.

Adds two things the old pipeline lacked:
  * per-author PER-ERA token cap, so a prolific author / collected-works omnibus
    (e.g. রবীন্দ্র রচনাবলী = 28% of pre_1950) cannot dominate an era's embedding.
    Cap = --cap-frac of the era's UNCAPPED cleaned total (default 0.10). Authors are
    de-duplicated to the PERSON (রচনাবলী/সমগ্র wrappers stripped), works included
    oldest-first and truncated at a sentence boundary when the budget runs out.
  * memoized + stdout-silenced stemming (the stemmer is chatty and slow per call).

Input : epub_dataset/<era>/*.txt  (the dataset built by build_epub_dataset.py)
Output: data_cleaned_v2/<era>.txt        # drop-in for src/  (' <s>\n' format, stemmed)
        data_cleaned_v2/stats.json        # per-era + per-author before/after counts

Run:  python clean_corpus_v2.py            # preview (no write)
      python clean_corpus_v2.py --write     # materialize data_cleaned_v2/
"""
import argparse
import contextlib
import csv
import json
import os
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATASET = ROOT / "epub_dataset"
OUTDIR = ROOT.parent / "data_cleaned_v2"
MANIFEST = DATASET / "manifest.csv"
ERAS = ["pre_1950", "1950_1970", "1970_1990", "1990_2010", "2010_2025"]
NEWS_DIR = ROOT / "data_raw" / "2010_2025"   # clean born-digital news (Prothom Alo)
NEWS_GLOB = "prothomalo_*.txt"               # the *_djvu.txt.txt files are OCR -> excluded
NEWS_ERA = "2010_2025"
nfc = lambda s: unicodedata.normalize("NFC", s or "")

# --- stemmer (memoized, silenced) -------------------------------------------
_devnull = open(os.devnull, "w")
with contextlib.redirect_stdout(_devnull):
    from bangla_stemmer.stemmer import stemmer
    _STEMMER = stemmer.BanglaStemmer()
_STEM = {}


def stem(tok):
    s = _STEM.get(tok)
    if s is None:
        with contextlib.redirect_stdout(_devnull):
            try:
                s = _STEMMER.stem(tok) or tok
            except Exception:
                s = tok
        _STEM[tok] = s
    return s


# --- token validity (identical rules to the original pipeline) --------------
BENGALI_VOWELS = set("ািীুূেৈোৌঅআইঈউঊএঐওঔঋৠঌৡ")
BANGLA_DIGITS = "০১২৩৪৫৬৭৮৯"

# Standard Bangla stopword list (stemmed to match the corpus),
# so filtering matches the original corpus exactly.
BENGALI_STOP_WORDS = set("""অতএব অথচ অথবা অনুযায়ী অনেক অনেকে অনেকেই অন্তত অন্য অবধি অবশ্য অর্থাৎ অর্থাত
আই আগামী আগে আগেই আছে আজ আদ্যভাগে আপনার আপনি আবার আমরা আমাকে আমাদের আমার আমি আর আরও ই ইত্যাদি ইহা উচিত
উত্তর উনি উপর উপরে এ এঁদের এঁরা এই একই একটি একটা একবার একে এক্ এখন এখনও এখানে এখানেই এটা এটাই এটি এত এতটাই এতে
এদের এব এবং এবার এমন এমনকী এমনি এর এরা এল এস এসে ঐ ও ওঁদের ওঁর ওঁরা ওই ওকে ওখানে ওদের ওর ওরা কখনও কত কবে কমনে
কয়েক কয়েকটি করছে করছেন করতে করবে করবেন করল করলে করলেন করা করাই করানো করায় করার করি করিতে করিয়া করিয়ে করে করেই
করেছিলেন করেছে করেছেন করেন করেনি কাউকে কাছ কাছে কাজ কাজে কারও কারণ কি কিংবা কিছু কিছুই কিন্তু কী কে কেউ কেউই কেন
কোটি কোন কোনও কোনো ক্ষেত্রে কিভাবে কিছুক্ষণ খুব গিয়ে গিয়েছে গুলি গেছে গেল গেলে গোটা চলে চান চায় চার চালু চেয়ে চেষ্টা
ছাড়া ছাড়াও ছিল ছিলাম ছিলেন ছিলো জন জনকে জনের জন্য জন্যে জানতে জানা জানানো জানায় জানিয়ে জানিয়েছে জানে জানেন জে টা টি ঠিক
তখন তত তথা তবু তবে তা তাঁকে তাঁদের তাঁর তাঁরা তাই তাও তাকে তাতে তাদের তার তারপর তারা তাহলে তাহা তাহাতে তাহার তিন তিনি
তিনিও তুমি তুলে তেমন তো তোমার তোমরা থাকবে থাকবেন থাকা থাকায় থাকে থাকেন থেকে থেকেই থেকেও দিকে দিতে দিন দিয়ে দিয়েছে দিয়েছেন
দিলেন দু দুই দুজন দুটি দুটো দেওয়া দেওয়ার দেখতে দেখা দেখে দেন দেয় দ্বারা ধরা ধরে নতুন নয় না নাই নাকি নাগাদ নানা নিজে নিজেই
নিজেদের নিজের নিতে নিয়ে নেই নেওয়া নেওয়ার পক্ষে পর পরে পরেই পরেও পর্যন্ত পাওয়া পারি পারে পারেন পি পেয়ে প্রতি প্রথম প্রভৃতি
প্রায় ফলে ফিরে ফের বক্তব্য বদলে বরং বলতে বলল বললেন বলা বলে বলেছেন বলেন বসে বহু বা বাদে বার বি বিনা বিভিন্ন বিশেষ বিষয়টি বেশ
বেশি ব্যবহার ব্যাপারে ভাবে ভাবেই ভাল মতো মতোই মধ্যভাগে মধ্যে মধ্যেই মধ্যেও মনে মাত্র মাধ্যমে মোট মোটেই যখন যত যতটা যথেষ্ট যদি
যদিও যা যাঁর যাঁরা যাওয়া যাওয়ার যাকে যাচ্ছে যাতে যাদের যান যাবে যায় যার যারা যিনি যে যেখানে যেতে যেন যেমন র রকম রয়েছে রাখা রেখে
লক্ষ শুধু শুরু সঙ্গে সঙ্গেও সব সবার সমস্ত সম্প্রতি সহ সহিত সাধারণ সামনে সি সুতরাং সে সেই সেখান সেখানে সেটা সেটাই সেটাও সেটি স্পষ্ট
স্বয়ং হইতে হইবে হইয়া হওয়া হওয়ায় হওয়ার হচ্ছে হত হতে হতেই হন হবে হবেন হয় হয়তো হয়নি হয়ে হয়েই হয়েছিল হয়েছে হয়েছেন হল হলে
হলেই হলেও হলো হাজার হিসাবে হিসেবে হোক এক থে আমা যা হা তাঁ ঢাকা বাংলাদেশ ভারত একজন হাত মুখ চোখ দিন বছর কথা সময়""".split())

_STOP = set()
for _w in BENGALI_STOP_WORDS:
    _STOP.add(stem(_w) or _w)


def is_valid(tok):
    if len(tok) <= 1 or len(tok) > 12:
        return False
    if all(c == tok[0] for c in tok):
        return False
    if all(c in BANGLA_DIGITS for c in tok):
        return False
    if not any(c in BENGALI_VOWELS for c in tok):
        return False
    if tok in _STOP:
        return False
    return True


# --- cleaning (identical regex chain to the original) -----------------------
RE_QUOTE = re.compile(r'([।?!])["”\'‘’]+')
RE_ENNUM = re.compile(r"[A-Za-z0-9]")
RE_PUNCT = re.compile(r"[“”‘’\"'.,()\[\]{};:@#$%^&*_+=~<>\\/—–\-]")
RE_WS = re.compile(r"\s+")
RE_SENT = re.compile(r"[|।?!]")


def clean_to_sentences(text):
    """Return list of cleaned, stemmed, stopword-filtered sentences (token strings)."""
    t = text.replace("\n", " ")
    t = RE_QUOTE.sub(r"\1", t)
    t = RE_ENNUM.sub("", t)
    t = RE_PUNCT.sub("", t)
    t = RE_WS.sub(" ", t)
    out = []
    for sent in RE_SENT.split(t):
        sent = sent.strip()
        if not sent:
            continue
        toks = [stem(w) for w in sent.split()]
        toks = [w for w in toks if is_valid(w)]
        if toks:
            out.append(" ".join(toks))
    return out


# --- author identity (collapse রচনাবলী/সমগ্র wrappers to the person) --------
COLL_WORD = re.compile(r"রচনাবলী|রচনাবলি|রচনাসমগ্র|রচনা ?সংকলন|রচনা ?সংগ্রহ|সমগ্র|গ্রন্থাবলী|গ্রন্থাবলি|গল্পসমগ্র|উপন্যাস ?সমগ্র")
NORM_BAD = re.compile(r"[^\wঀ-৿]+")


def author_key(a):
    a = nfc(a)
    if "–" in a or "—" in a:
        a = re.split(r"[–—]", a)[-1]
    a = COLL_WORD.sub("", a)
    return NORM_BAD.sub(" ", a.lower()).strip() or "(unknown)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--cap-frac", type=float, default=0.10,
                    help="per-author cap as a fraction of each era's uncapped cleaned tokens")
    ap.add_argument("--news", action="store_true",
                    help="fold the clean Prothom Alo news into 2010_2025 (uncapped many-author register)")
    args = ap.parse_args()

    rows = list(csv.DictReader(open(MANIFEST, encoding="utf-8-sig")))
    print(f"cleaning {len(rows)} dataset files (stemming is memoized; first pass is the slow one) ...",
          file=sys.stderr)

    # PASS 1: clean every doc, record (era, author, year, sentences, ntok)
    docs = []
    era_raw = Counter()
    for i, r in enumerate(rows):
        if i % 500 == 0:
            print(f"  cleaned {i}/{len(rows)} ...", file=sys.stderr)
        fp = DATASET / r["path"]
        try:
            text = fp.read_text(encoding="utf-8")
        except Exception:
            continue
        sents = clean_to_sentences(text)
        ntok = sum(s.count(" ") + 1 for s in sents if s)
        if ntok == 0:
            continue
        era = r["era"]
        docs.append({"era": era, "ak": author_key(r["author"]),
                     "year": int(r["year"]), "sents": sents, "ntok": ntok})
        era_raw[era] += ntok

    # cap budget per era
    cap = {e: int(era_raw[e] * args.cap_frac) for e in era_raw}

    # PASS 2: per era, per author (oldest-first), include up to the cap (sentence-truncated)
    kept_sents = defaultdict(list)        # era -> [sentence, ...]
    era_kept = Counter()
    auth_kept = defaultdict(Counter)      # era -> author -> kept tokens
    auth_raw = defaultdict(Counter)
    by_era_auth = defaultdict(lambda: defaultdict(list))
    for d in docs:
        by_era_auth[d["era"]][d["ak"]].append(d)
        auth_raw[d["era"]][d["ak"]] += d["ntok"]

    for era in ERAS:
        budget_cap = cap.get(era, 0)
        for ak, dlist in by_era_auth[era].items():
            budget = budget_cap
            for d in sorted(dlist, key=lambda x: (x["year"],)):
                if budget <= 0:
                    break
                for s in d["sents"]:
                    n = s.count(" ") + 1
                    if budget <= 0:
                        break
                    kept_sents[era].append(s)
                    era_kept[era] += n
                    auth_kept[era][ak] += n
                    budget -= n

    # fold in clean news (uncapped) -> 2010_2025
    news_tok = 0
    news_files = 0
    if args.news:
        for fp in sorted(NEWS_DIR.glob(NEWS_GLOB)):
            try:
                sents = clean_to_sentences(nfc(fp.read_text(encoding="utf-8")))
            except Exception:
                continue
            news_files += 1
            for s in sents:
                kept_sents[NEWS_ERA].append(s)
                n = s.count(" ") + 1
                era_kept[NEWS_ERA] += n
                news_tok += n
        print(f"folded {news_files} news files into {NEWS_ERA}: +{news_tok:,} clean tokens "
              f"({news_tok*100//max(era_kept[NEWS_ERA],1)}% of that era)", file=sys.stderr)

    # --- report ---
    print(f"\nunique stems cached: {len(_STEM):,}")
    if args.news:
        print(f"news folded into {NEWS_ERA}: +{news_tok:,} tokens "
              f"({news_tok*100//max(era_kept[NEWS_ERA],1)}% of the era)")
    print(f"\n  {'era':10} {'docs':>5} {'raw_clean':>12} {'cap/auth':>11} {'after_cap':>12} {'topauthor%':>11}")
    total_after = 0
    for era in ERAS:
        if not era_raw.get(era):
            continue
        top = auth_kept[era].most_common(1)
        toppct = (top[0][1] * 100 / era_kept[era]) if (top and era_kept[era]) else 0
        total_after += era_kept[era]
        ndoc = sum(len(v) for v in by_era_auth[era].values())
        print(f"  {era:10} {ndoc:>5} {era_raw[era]:>12,} {cap[era]:>11,} {era_kept[era]:>12,} {toppct:>10.1f}%")
    print(f"  {'TOTAL':10} {len(docs):>5} {sum(era_raw.values()):>12,} {'':>11} {total_after:>12,}")
    print(f"\n  (top author per era is now <= cap-frac = {args.cap_frac:.0%} of the uncapped era)")

    if not args.write:
        print("\n(preview only — re-run with --write to materialize data_cleaned_v2/)")
        return

    OUTDIR.mkdir(parents=True, exist_ok=True)
    for era in ERAS:
        if kept_sents.get(era):
            (OUTDIR / f"{era}.txt").write_text(" <s>\n".join(kept_sents[era]), encoding="utf-8")
    stats = {era: {"docs": sum(len(v) for v in by_era_auth[era].values()),
                   "uncapped_tokens": era_raw[era], "cap_per_author": cap.get(era, 0),
                   "final_tokens": era_kept[era],
                   "top_authors_after": auth_kept[era].most_common(8)}
             for era in ERAS if era_raw.get(era)}
    if args.news:
        stats[NEWS_ERA]["news_tokens_added"] = news_tok
        stats[NEWS_ERA]["news_files"] = news_files
    (OUTDIR / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwrote {len([e for e in ERAS if kept_sents.get(e)])} era files + stats.json -> {OUTDIR}")


if __name__ == "__main__":
    main()
