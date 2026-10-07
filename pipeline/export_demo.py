"""Assemble within_data.json from REAL pipeline outputs and inject it into the page.

Nothing here invents values: every number comes from the dataset files or from the
out-of-fold predictions written by refed_train.py / tufts_train.py.

Replay selection is fixed BEFORE looking at the outcome: by default subject 7, and the
first REFED clip the authors designed as high-valence / high-arousal (video 4). Whatever the
model predicted for that held-out recording is what the reveal shows - match or not.

    python export_demo.py --refed-root ./REFED-dataset --refed-out ./out \
        --tufts-windows <csv dir> --tufts-out ./out \
        --template ../within.html --html-out ../within.live.html
"""
import argparse, glob, json, os, re
import numpy as np
from common import load_mat as loadmat, mat_keys
from scipy.signal import resample_poly

from refed_train import preprocess, FS, SIGNALS, FS_RAW, subject_ids
from tufts_train import COLS, subject_from_path
from common import to_class3, refed_label_path

MARK = "window.WITHIN_DATA = null;"
# REFED video design categories from the official repo (0 neutral, 1 happy, 2 fearful, 3 sad, 4 relaxed)
VIDEO_DESIGN = [0, 3, 2, 1, 4, 3, 1, 4, 2, 0, 2, 0, 3, 4, 1]


def r(a, n=3):
    return [round(float(v), n) for v in np.asarray(a).ravel()]


def z(a):
    a = np.asarray(a, float); return (a - a.mean()) / (a.std() + 1e-9)


def refed_block(a):
    res = json.load(open(os.path.join(a.refed_out, "refed_results.json")))
    oof = np.load(os.path.join(a.refed_out, "refed_oof.npz"))
    sid = str(a.subject)
    if not os.path.exists(os.path.join(a.refed_root, "data", sid, "fNIRS_videos.mat")):
        fallback = subject_ids(a.refed_root)[0]   # pre-specified rule, independent of any result
        print(f"[export] subject {sid} not found; using the lowest-numbered subject {fallback}")
        sid = fallback
    video = a.video or (VIDEO_DESIGN.index(1) + 1)
    raw = loadmat(os.path.join(a.refed_root, "data", sid, "fNIRS_videos.mat"))[f"video_{video}"]
    x = preprocess(raw)                                    # (6, 51, T) @ 10 Hz
    chan_mean = x.mean(1)                                  # (6, T)
    # Sanity check on REFED channel order (only HbT at index 2 is confirmed by the official code):
    # HbT should track HbO + HbR, and HbO usually varies more than HbR.
    raw_mean = preprocess(raw).mean(1)
    hbt_fit = float(np.corrcoef(raw_mean[2], raw_mean[0] + raw_mean[1])[0, 1])
    hbo_gt_hbr = bool(raw_mean[0].std() > raw_mean[1].std())
    channel_check = {"corr_hbt_vs_hbo_plus_hbr": round(hbt_fit, 3), "hbo_varies_more_than_hbr": hbo_gt_hbr,
                     "passed": bool(hbt_fit > 0.9 and hbo_gt_hbr)}
    if not channel_check["passed"]:
        print(f"[export] WARNING: REFED channel order check failed {channel_check}; verify SIGNALS order before naming traces")
    disp = resample_poly(chan_mean, up=2, down=5, axis=-1)  # -> 4 Hz for the browser
    m = (oof["subject"] == int(sid)) & (oof["video"] == int(video))
    if not m.any():
        raise SystemExit(f"no out-of-fold windows for subject {sid} video {video}")
    t = oof["t"][m]; o = np.argsort(t); t = t[o]
    pa = oof[f"p_{a.model}_arousal"][m][o]; pv = oof[f"p_{a.model}_valence"][m][o]
    ca = oof["c_arousal"][m][o]; cv = oof["c_valence"][m][o]
    lab = loadmat(refed_label_path(a.refed_root, sid))[f"video_{video}"]
    lab = (lab.astype(float) - 1) / 254
    dur = raw.shape[-1] / FS_RAW
    model_cls = {"arousal": int(pa.mean(0).argmax()), "valence": int(pv.mean(0).argmax())}
    report_cls = {"arousal": int(to_class3(lab[:, 1].mean())), "valence": int(to_class3(lab[:, 0].mean()))}
    stats_subjects = len(subject_ids(a.refed_root))
    n_rec = sum(sum(1 for name in mat_keys(os.path.join(a.refed_root, "data", s, "fNIRS_videos.mat"))
                    if name.startswith("video_")) for s in subject_ids(a.refed_root))
    return {
        "stats": {"subjects": stats_subjects, "recordings": int(n_rec), "fs": round(FS_RAW, 2), "signal_types": len(SIGNALS)},
        "replay": {
            "subject": int(sid), "video": int(video), "duration": round(dur, 2), "fs": 4,
            "hbo": r(z(disp[0])), "hbr": r(z(disp[1])), "hbt": r(z(disp[2])),
            "pred_t": r(t, 2),
            "pred_arousal": r(0.5 + 0.5 * (pa[:, 2] - pa[:, 0])),   # 0..1 display scale
            "pred_valence": r(pv[:, 2] - pv[:, 0]),                  # -1..1 display scale
            "report_t": r(np.linspace(0, dur, len(lab)), 2),
            "report_arousal": r(lab[:, 1]), "report_valence": r(lab[:, 0] * 2 - 1),
            "model_class": model_cls, "report_class": report_cls,
            "channel_check": channel_check,
        },
        "metrics": {"protocol": res["protocol"], "demo_model": a.model, "models": res["models"],
                    "subject_video_holdout": res.get("subject_video_holdout")},
    }


def tufts_block(a):
    res = json.load(open(os.path.join(a.tufts_out, "tufts_results.json")))
    oof = np.load(os.path.join(a.tufts_out, "tufts_oof.npz"))
    sid = a.tufts_subject or int(oof["subject"][0])
    path = [p for p in glob.glob(os.path.join(a.tufts_windows, "*.csv")) if subject_from_path(p) == sid][0]
    import pandas as pd
    df = pd.read_csv(path)
    # Rebuild a continuous trace: each 0.6 s-stride window contributes its last 3 samples.
    # first window contributes all its samples, every later window its 3 new ones -> no missing first ~29 s
    first = df[df.chunk == df.chunk.min()]
    tail = pd.concat([first.iloc[:-3], df.groupby("chunk").tail(3)])
    hbo = tail[["AB_I_O", "CD_I_O"]].mean(axis=1).values; hbr = tail[["AB_I_DO", "CD_I_DO"]].mean(axis=1).values
    lab_per_sample = tail.label.values
    # first occurrence of 0,1,2,3 as contiguous blocks, in order
    runs, start = [], 0
    for i in range(1, len(lab_per_sample) + 1):
        if i == len(lab_per_sample) or lab_per_sample[i] != lab_per_sample[start]:
            runs.append((int(lab_per_sample[start]), start, i)); start = i
    chosen, want = [], 0
    for lvl, s0, s1 in runs:
        if lvl == want:
            chosen.append((lvl, s0, s1)); want += 1
        if want == 4:
            break
    m = oof["subject"] == sid
    ch = oof["chunk"][m]; p = oof[f"p_{a.tufts_model}"][m]
    load = 100 * (p * np.arange(4)).sum(1) / 3                 # expected level, rescaled 0-100
    seg_hbo, seg_hbr, blocks, lt, lv, cursor = [], [], [], [], [], 0
    for lvl, s0, s1 in chosen:
        seg_hbo += list(hbo[s0:s1]); seg_hbr += list(hbr[s0:s1])
        lead = len(first) - 3                                      # samples before chunk 0's last 3
        c0, c1 = max(0, (s0 - lead) // 3), max(0, (s1 - lead) // 3)  # sample index -> chunk index
        sel = (ch >= c0) & (ch < c1)
        lt += list(cursor + np.maximum(0, (ch[sel] - c0) * 3 + (lead if c0 == 0 else 0) - (s0 if c0 == 0 else 0))); lv += list(load[sel])
        blocks.append({"level": lvl, "start": cursor, "end": cursor + (s1 - s0)})
        cursor += s1 - s0
    return {
        "stats": {"subjects": res["n_subjects"], "fs": 5.2, "levels": 4, "window_s": 30},
        "replay": {"subject": int(sid), "fs": 5.2, "hbo": r(z(seg_hbo)), "hbr": r(z(seg_hbr)),
                   "blocks": blocks, "load_t": r(lt, 1), "load": r(lv, 1)},
        "metrics": {"protocol": res["protocol"], "demo_model": a.tufts_model, "models": res["models"]},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refed-root"); ap.add_argument("--refed-out")
    ap.add_argument("--tufts-windows"); ap.add_argument("--tufts-out")
    ap.add_argument("--subject", default="7"); ap.add_argument("--video", type=int, default=None)
    ap.add_argument("--model", default="xgb", help="REFED model shown in the replay (choose before looking)")
    ap.add_argument("--tufts-subject", type=int, default=None); ap.add_argument("--tufts-model", default="xgb")
    ap.add_argument("--template", required=True); ap.add_argument("--html-out", required=True)
    ap.add_argument("--json-out", default="within_data.json")
    ap.add_argument("--physionet-out", default=None, help="folder with physionet_results.json (optional)")
    ap.add_argument("--personalize-out", default=None, help="folder with personalization_results.json (optional)")
    a = ap.parse_args()
    # Each block is optional: sections without real data stay in labelled preview mode on the page.
    data = {"meta": {"source": "pipeline", "synthetic": bool(os.environ.get("WITHIN_SMOKE_TEST"))}}
    if a.refed_root and a.refed_out:
        data["refed"] = refed_block(a)
    if a.tufts_windows and a.tufts_out:
        data["tufts"] = tufts_block(a)
    if a.physionet_out:
        data["physionet"] = json.load(open(os.path.join(a.physionet_out, "physionet_results.json")))
    if a.personalize_out:
        data["personalization"] = json.load(open(os.path.join(a.personalize_out, "personalization_results.json")))
    json.dump(data, open(a.json_out, "w"))
    html = open(a.template).read()
    assert MARK in html, "data marker not found in template"
    open(a.html_out, "w").write(html.replace(MARK, "window.WITHIN_DATA = " + json.dumps(data) + ";"))
    print("wrote", a.json_out, "and", a.html_out)


if __name__ == "__main__":
    main()
