"""PhysioNet 'mental-fnirs' v1.0: fNIRS + TCD during the same n-back workload task.

Temple-relevance validation for THINK. Not a classifier: it measures whether workload
changes BOTH cortical oxygenation (fNIRS HbO) and cerebral blood-flow velocity (TCD, MCA).

Data (open access, PhysioNet Contributor Review Health Data License 1.5.0):
    https://physionet.org/content/mental-fnirs/1.0/
    wget -r -N -c -np https://physionet.org/files/mental-fnirs/1.0/
Cite: Mukli, Yabluchanskiy & Csipo (2021), PhysioNet, doi:10.13026/zfb2-1g43

Protocol: 0-back, 1-back, 0-back, 2-back, ~134 s each. 14 participants recruited.
Both modalities use the same contrast the dataset authors use for TCD:
    response[n-back] = signal[n-back] - signal[preceding 0-back]
    window = 10 s to 110 s after session onset (first 10 s discarded, 100 s window)
TCD:   % change in mean MCA velocity, from the authors' Summary-statistics_MCA.csv (L/R average).
fNIRS: change in mean prefrontal HbO (channels 1-24: F3, AF7, AF3, Fz, Fpz, AF4, F4, AF8 sources),
       from raw 760/850 nm intensities via the modified Beer-Lambert law, expressed in units of the
       participant's own 0-back HbO standard deviation (so participants are comparable).

    python physionet_adapter.py --root ./physionet.org/files/mental-fnirs/1.0 --out ./out
"""
import argparse, glob, os, re
import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt, detrend
from scipy.stats import wilcoxon

from common import dump

# ANALYSIS ASSUMPTIONS (prototype choices, not dataset ground truth; a reviewer may reasonably change them):
#   extinction coefficients (Prahl), DPF 6.0, 3 cm source-detector distance, 0.005-0.1 Hz band-pass,
#   prefrontal ROI = channels 1-24. Results are reported in SD units partly to reduce dependence on DPF.
# Molar extinction coefficients (cm^-1 / M), Prahl compilation: rows = 760, 850 nm; cols = HbO, HbR
EXT = np.array([[1486.0, 3843.707], [2526.0, 1798.643]])
DPF, DIST_CM = 6.0, 3.0
WIN = (10.0, 110.0)
PFC_CHANNELS = np.arange(24)          # channels 1-24 (0-based), prefrontal sources S1-S8
RECRUITED = 14


def parse_hdr(path):
    txt = open(path, errors="ignore").read()
    fs = float(re.search(r"SamplingRate=([\d.]+)", txt).group(1))
    ev_block = re.search(r'Events="#(.*?)#"', txt, re.S).group(1)
    events = [tuple(map(float, l.split()[:3])) for l in ev_block.strip().splitlines() if l.strip()]
    mask_block = re.search(r'S-D-Mask="#(.*?)#"', txt, re.S).group(1)
    mask = np.array([[int(v) for v in l.split()] for l in mask_block.strip().splitlines() if l.strip()])
    return fs, sorted(events), mask


def hbo_roi(folder):
    hdr = glob.glob(os.path.join(folder, "*.hdr"))[0]
    base = hdr[:-4]
    fs, events, mask = parse_hdr(hdr)
    cols = np.flatnonzero(mask.ravel())            # row-major S-D order == published channel order
    assert len(cols) == 48, f"{folder}: expected 48 channels, mask has {len(cols)}"
    I = [np.loadtxt(base + ext)[:, cols] for ext in (".wl1", ".wl2")]
    I = [np.clip(x, 1e-6, None) for x in I]
    # base-10 optical density: Prahl extinction coefficients are decadic (cm^-1 M^-1)
    od = np.stack([-np.log10(x / x.mean(0, keepdims=True)) for x in I])     # (2, T, 48)
    sos = butter(3, [0.005, 0.1], btype="band", fs=fs, output="sos")
    od = sosfiltfilt(sos, detrend(od, axis=1), axis=1)
    inv = np.linalg.pinv(EXT * DPF * DIST_CM)                                # (2 chromophores, 2 wavelengths)
    conc = np.einsum("cw,wtk->ctk", inv, od) * 1e6                           # micromolar
    roi = conc[0][:, PFC_CHANNELS].mean(1)                                   # HbO, prefrontal ROI
    print(f"[fNIRS] {os.path.basename(folder)} markers (time s, code): {[(round(e[0], 1), int(e[1])) for e in events]}", flush=True)
    if len(events) != 4:
        print(f"[fNIRS] WARNING {os.path.basename(folder)}: expected 4 session markers (0-back, 1-back, 0-back, 2-back), found {len(events)}; using the first 4")
    onsets = [e[0] for e in events][:4]
    if len(onsets) < 4:
        raise ValueError(f"{folder}: expected 4 session markers, found {len(events)}")
    seg = lambda t0: roi[int((t0 + WIN[0]) * fs): int((t0 + WIN[1]) * fs)]
    s0a, s1, s0b, s2 = (seg(t) for t in onsets)
    sd0 = np.concatenate([s0a, s0b]).std() + 1e-9
    return {"d1": (s1.mean() - s0a.mean()) / sd0, "d2": (s2.mean() - s0b.mean()) / sd0, "fs": fs}


def summarise(x):
    x = np.asarray(x, float)
    out = {"mean": float(x.mean()), "sem": float(x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else 0.0,
           "n": int(len(x)), "n_positive": int((x > 0).sum())}
    if len(x) >= 6:
        out["p_wilcoxon_greater"] = float(wilcoxon(x, alternative="greater").pvalue)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="folder containing NIRS_data/ and TCD_data/")
    ap.add_argument("--out", default="./out")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    tcd = pd.read_csv(os.path.join(a.root, "TCD_data", "Summary-statistics_MCA.csv"), encoding="utf-8-sig")
    col = lambda k: tcd[f"{k} Average Blood Flow Velocity"].astype(float)
    ok = col("0-back(1)").notna() & col("1-back").notna() & col("0-back(2)").notna() & col("2-back").notna()
    z1, o1, z2, t2 = (col(k)[ok].values for k in ("0-back(1)", "1-back", "0-back(2)", "2-back"))
    tcd_d1, tcd_d2 = (o1 - z1) / z1 * 100, (t2 - z2) / z2 * 100

    f1, f2, used, failed = [], [], [], {}
    for folder in sorted(glob.glob(os.path.join(a.root, "NIRS_data", "SUBJ*"))):
        try:
            r = hbo_roi(folder); f1.append(r["d1"]); f2.append(r["d2"]); used.append(os.path.basename(folder))
            print(f"[fNIRS] {os.path.basename(folder)}  1-back {r['d1']:+.2f} SD  2-back {r['d2']:+.2f} SD", flush=True)
        except Exception as e:                     # report, never silently drop
            failed[os.path.basename(folder)] = str(e); print("[fNIRS] skip", folder, e)
    tcd_files = sorted(os.path.basename(p)[:-4] for p in glob.glob(os.path.join(a.root, "TCD_data", "SUBJ*.csv")))

    res = {
        "dataset": "PhysioNet mental-fnirs v1.0 (Mukli, Yabluchanskiy & Csipo, 2021)",
        "doi": "10.13026/zfb2-1g43",
        "recruited": RECRUITED,
        "n_nirs_recordings": len(glob.glob(os.path.join(a.root, "NIRS_data", "SUBJ*"))),
        "n_tcd_recordings": len(tcd_files),
        "contrast": "n-back minus preceding 0-back, 10-110 s window",
        "tcd": {"unit": "% change in mean MCA velocity (L/R average)", "source": "Summary-statistics_MCA.csv",
                "one_back": summarise(tcd_d1), "two_back": summarise(tcd_d2)},
        "fnirs": ({"unit": "change in prefrontal HbO, in SD of the participant's 0-back signal",
                   "subjects": used, "failed": failed,
                   "one_back": summarise(f1), "two_back": summarise(f2)} if f1 else None),
    }
    dump(res, os.path.join(a.out, "physionet_results.json"))
    print("[done]", os.path.join(a.out, "physionet_results.json"))


if __name__ == "__main__":
    main()
