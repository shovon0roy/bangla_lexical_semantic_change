"""Stage 8: the frequency / n-gram prong — track per-era word frequency to catch
EMERGENT and DECLINING words (the signal the embedding-shift prong misses, e.g. words
too rare in the early era to score as 'shifters').

For each era (chronological): normalized frequency per million tokens of every word.
  emergent  : near-absent early, frequent late  (e.g. ভাইরাল, ইন্টারনেট, করোনা)
  declining : frequent early, fading late
Also writes trajectories for a set of showcase words.

Deterministic (pure counting). Reads data_cleaned_v2/<era>.txt.

Output: results_v2/freq_by_era.json        (per-era per-million freq, all words >= MINTOTAL)
        results_v2/emergent_words.csv / declining_words.csv
        results_v2/freq_trajectories.csv    (showcase words)
        results_v2/figure_frequency_trends.png

Run:  .venv/bin/python src/N-gram/freq_trends_v2.py
"""
import csv
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data_cleaned_v2"
OUT = ROOT / "results_v2"
ERAS = ["pre_1950", "1950_1970", "1970_1990", "1990_2010", "2010_2025"]
MINTOTAL = 30          # ignore very rare words (noise)
SHOWCASE = ["ডিজিটাল", "ইন্টারনেট", "মোবাইল", "কম্পিউটার", "অনলাইন", "ভাইরাল",
            "ফেসবুক", "করোনা", "গাজা", "গুম", "স্বাধীনতা", "পরিবেশ"]


def era_counts(era):
    toks = (DATA / f"{era}.txt").read_text(encoding="utf-8").split()
    toks = [t for t in toks if t != "<s>" and t != "<s/>"]
    return Counter(toks), len(toks)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    counts, totals = {}, {}
    for e in ERAS:
        counts[e], totals[e] = era_counts(e)
        print(f"  {e:10} tokens={totals[e]:>10,} types={len(counts[e]):>8,}")

    vocab = set()
    for e in ERAS:
        vocab |= {w for w, c in counts[e].items() if c >= 5}
    # per-million freq per era; keep words with decent total
    pm = {}
    for w in vocab:
        tot = sum(counts[e][w] for e in ERAS)
        if tot < MINTOTAL:
            continue
        pm[w] = {e: round(counts[e][w] / totals[e] * 1e6, 3) for e in ERAS}

    (OUT / "freq_by_era.json").write_text(json.dumps(pm, ensure_ascii=False), encoding="utf-8")
    print(f"\nwords tracked (total>={MINTOTAL}): {len(pm):,}")

    early_eras = ["pre_1950", "1950_1970", "1970_1990"]
    late = "2010_2025"

    def early_avg(w):
        return sum(pm[w][e] for e in early_eras) / len(early_eras)

    rows = [(w, early_avg(w), pm[w][late], pm[w][late] - early_avg(w)) for w in pm]
    emergent = sorted(rows, key=lambda r: r[3], reverse=True)
    declining = sorted(rows, key=lambda r: r[3])

    def dump(path, data):
        with open(path, "w", encoding="utf-8-sig", newline="") as f:
            wr = csv.writer(f)
            wr.writerow(["word", "early_avg_pm", "y2010_2025_pm", "delta_pm"] + ERAS)
            for w, ea, lt, d in data:
                wr.writerow([w, round(ea, 2), round(lt, 2), round(d, 2)] + [pm[w][e] for e in ERAS])

    dump(OUT / "emergent_words.csv", emergent[:200])
    dump(OUT / "declining_words.csv", declining[:200])

    print("\n=== TOP EMERGENT (near-absent early -> frequent 2010-25) ===")
    shown = 0
    for w, ea, lt, d in emergent:
        if ea < 2.0 and shown < 25:      # genuinely new: tiny early presence
            print(f"  {w:16} {ea:7.2f} -> {lt:8.2f}  (+{d:.1f}/M)")
            shown += 1
    print("\n=== TOP DECLINING ===")
    for w, ea, lt, d in declining[:15]:
        print(f"  {w:16} {ea:7.2f} -> {lt:8.2f}  ({d:.1f}/M)")

    # showcase trajectories
    with open(OUT / "freq_trajectories.csv", "w", encoding="utf-8-sig", newline="") as f:
        wr = csv.writer(f); wr.writerow(["word"] + ERAS)
        traj = {}
        for w in SHOWCASE:
            if w in pm:
                wr.writerow([w] + [pm[w][e] for e in ERAS]); traj[w] = [pm[w][e] for e in ERAS]
            else:
                wr.writerow([w] + ["NA"] * len(ERAS))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib import font_manager
        for cand in ["Noto Sans Bengali", "Kalpurush", "Siyam Rupali"]:
            if any(cand in f.name for f in font_manager.fontManager.ttflist):
                plt.rcParams["font.family"] = cand
                break
        fig, ax = plt.subplots(figsize=(9, 5.5))
        xs = list(range(len(ERAS)))
        for w, ys in traj.items():
            ax.plot(xs, ys, marker="o", label=w)
        ax.set_xticks(xs); ax.set_xticklabels(["pre1950", "1950-70", "1970-90", "1990-2010", "2010-25"], rotation=20)
        ax.set_ylabel("frequency per million"); ax.set_title("frequency trajectories")
        ax.legend(ncol=2, fontsize=8); fig.tight_layout()
        fig.savefig(OUT / "figure_frequency_trends.png", dpi=150); plt.close(fig)
        print("\n  saved figure_frequency_trends.png")
    except Exception as e:
        print(f"\n  (figure skipped: {e})")

    print(f"\nwrote freq_by_era.json, emergent/declining/ trajectories -> {OUT}")


if __name__ == "__main__":
    main()
