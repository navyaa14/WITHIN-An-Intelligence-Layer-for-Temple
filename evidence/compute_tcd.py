"""Reproduce the PhysioNet TCD evidence from the authors' published per-session summary.

Source: Mukli, Yabluchanskiy & Csipo (2021). Mental workload during n-back task: fNIRS and TCD.
PhysioNet, doi:10.13026/zfb2-1g43. File: TCD_data/Summary-statistics_MCA.csv
Contrast: % change in mean MCA blood-flow velocity (left/right average), n-back vs the 0-back block before it.

    python compute_tcd.py                       # downloads the summary file (no account needed)
    python compute_tcd.py --csv Summary-statistics_MCA.csv
"""
import argparse, json, os, urllib.request
import numpy as np, pandas as pd
from scipy.stats import wilcoxon

URL = "https://physionet.org/files/mental-fnirs/1.0/TCD_data/Summary-statistics_MCA.csv?download"

ap = argparse.ArgumentParser(); ap.add_argument("--csv", default="Summary-statistics_MCA.csv"); a = ap.parse_args()
if not os.path.exists(a.csv):
    urllib.request.urlretrieve(URL, a.csv)
d = pd.read_csv(a.csv, encoding="utf-8-sig")
v = lambda k: d[f"{k} Average Blood Flow Velocity"].astype(float).values
z1, o1, z2, t2 = v("0-back(1)"), v("1-back"), v("0-back(2)"), v("2-back")
one, two = (o1 - z1) / z1 * 100, (t2 - z2) / z2 * 100

def stats(x):
    return {"mean_pct": round(float(x.mean()), 3), "sem_pct": round(float(x.std(ddof=1) / np.sqrt(len(x))), 3),
            "median_pct": round(float(np.median(x)), 3), "n": int(len(x)), "n_increased": int((x > 0).sum()),
            "p_wilcoxon_one_sided": float(wilcoxon(x, alternative="greater").pvalue)}

res = {"source": "PhysioNet mental-fnirs v1.0, TCD_data/Summary-statistics_MCA.csv (Mukli, Yabluchanskiy & Csipo, 2021)",
       "doi": "10.13026/zfb2-1g43",
       "measure": "% change in mean middle-cerebral-artery blood-flow velocity (L/R average), n-back vs preceding 0-back",
       "one_back": stats(one), "two_back": stats(two),
       "two_back_vs_one_back_p_wilcoxon_one_sided": float(wilcoxon(two, one, alternative="greater").pvalue),
       "note": "TCD measures blood-flow velocity in a large artery. It is not Temple's Flow."}
json.dump(res, open("physionet_tcd_results.json", "w"), indent=1)
pd.DataFrame({"participant": d["Participant#"], "one_back_pct_change": one.round(3), "two_back_pct_change": two.round(3)}
             ).to_csv("physionet_tcd_per_participant.csv", index=False)

try:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    ink, ink2, rule, blue, paper = "#191917", "#55534D", "#CFCAC0", "#2F4A80", "#EFECE5"
    fig, ax = plt.subplots(figsize=(7, 5), dpi=200); fig.patch.set_facecolor(paper); ax.set_facecolor(paper)
    rng = np.random.default_rng(0)
    for i, (x, lab) in enumerate([(one, "1-back"), (two, "2-back")]):
        m, se = x.mean(), x.std(ddof=1) / np.sqrt(len(x))
        ax.bar(i, m, 0.55, color=blue, alpha=.9, zorder=2); ax.errorbar(i, m, se, color=ink, capsize=6, lw=1.2, zorder=3)
        ax.scatter(i + rng.uniform(-.14, .14, len(x)), x, s=16, color=ink, alpha=.55, zorder=4)
        ax.text(i, max(x.max(), m + se) + 1.2, f"{m:+.1f}%\n{(x > 0).sum()} of {len(x)} up", ha="center", va="bottom", color=ink, fontsize=10)
    ax.axhline(0, color=ink2, lw=1); ax.set_xticks([0, 1], ["1-back", "2-back"]); ax.set_xlim(-.6, 1.6)
    ax.set_ylabel("blood-flow velocity change vs 0-back (%)", color=ink2)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    for s in ("left", "bottom"): ax.spines[s].set_color(rule)
    ax.tick_params(colors=ink2); ax.set_ylim(min(-4, one.min() - 1), two.max() + 6)
    ax.set_title("Brain blood-flow velocity rises at high workload (TCD, n = 14)", color=ink, fontsize=12, loc="left")
    fig.text(.01, .01, "Mean ± SE, dots = participants. PhysioNet mental-fnirs v1.0 (Mukli et al., 2021). TCD is not Temple's Flow.",
             fontsize=7, color=ink2)
    fig.tight_layout(rect=(0, .03, 1, 1)); fig.savefig("physionet_tcd_chart.png", facecolor=paper)
except ImportError:
    pass
print(json.dumps(res, indent=1))
