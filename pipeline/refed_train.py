"""REFED (NeurIPS 2025): fNIRS-only valence / arousal, subject-independent.

Data:   https://huggingface.co/datasets/REFED2025/REFED-dataset  (CC-BY-NC-SA 4.0)
Layout expected (as in the official REFED-codes repo):
    <root>/data/<subject>/fNIRS_videos.mat      video_1..video_15, each (6, 51, T) @ 1000/21 Hz
    <root>/data/<subject>/fNIRS_baselines.mat   optional, same keys
    <root>/annotations/<subject>_label.mat      video_k: (L, 2) -> [valence, arousal], raw 1..255

Protocol:
  * 3 classes per dimension (low / medium / high, thresholds 0.4 / 0.6 on the 0-1 scale).
  * GroupKFold over SUBJECTS. Validation subjects for early stopping are carved out of
    the training subjects. No subject ever appears in two of {train, val, test}.
  * Every window gets an out-of-fold prediction, so any subject can be replayed in the demo.
  * Note: the published REFED benchmark is within-subject leave-one-trial-out.
    Our numbers answer a harder question and are NOT directly comparable.

    python refed_train.py --root ./REFED-dataset --out ./out
"""
import argparse, os, re, time
import numpy as np
from scipy.signal import butter, sosfiltfilt, resample_poly
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from xgboost import XGBClassifier

from common import load_mat as loadmat
from common import refed_label_path
from common import to_class3, window_features, score, summarise, softmax, assert_subject_disjoint, dump

FS_RAW = 1000 / 21          # 47.62 Hz
FS = 10                     # working rate after resampling
# Index 2 is HbT in the official feature notebook. Verify the remaining order
# against the dataset documentation before publishing per-signal claims.
SIGNALS = ["HbO", "HbR", "HbT", "780nm", "805nm", "830nm"]


def preprocess(x):
    """(6, 51, T) raw -> (6, 51, T') band-passed 0.01-0.5 Hz, resampled to 10 Hz."""
    sos = butter(3, [0.01, 0.5], btype="band", fs=FS_RAW, output="sos")
    x = sosfiltfilt(sos, x.astype(np.float64), axis=-1)
    return resample_poly(x, up=21, down=100, axis=-1).astype(np.float32)   # 47.62 * 21/100 = 10 Hz


def subject_ids(root):
    d = os.path.join(root, "data")
    return sorted([s for s in os.listdir(d) if os.path.isdir(os.path.join(d, s))], key=lambda s: int(re.sub(r"\D", "", s) or 0))


def load_subject(root, sid):
    m = loadmat(os.path.join(root, "data", sid, "fNIRS_videos.mat"))
    videos = {k: preprocess(m[f"video_{k}"]) for k in range(1, 16) if f"video_{k}" in m}
    bpath = os.path.join(root, "data", sid, "fNIRS_baselines.mat")
    if os.path.exists(bpath):
        b = loadmat(bpath)
        ref = np.concatenate([preprocess(b[k]) for k in b if k.startswith("video_")], -1)
        norm_src = "baseline"
    else:   # label-free fallback: the subject's own unlabeled recordings
        ref = np.concatenate(list(videos.values()), -1)
        norm_src = "subject-recordings"
    mu, sd = ref.mean(-1, keepdims=True), ref.std(-1, keepdims=True) + 1e-6
    videos = {k: (v - mu) / sd for k, v in videos.items()}
    lab = loadmat(refed_label_path(root, sid))
    labels = {k: (lab[f"video_{k}"].astype(float) - 1) / 254 for k in videos}   # -> 0..1, [:,0]=val [:,1]=aro
    return videos, labels, norm_src


def make_windows(videos, labels, win_s, stride_s):
    X, F, yv, ya, cv, ca, vid, tc = [], [], [], [], [], [], [], []
    W, S = int(win_s * FS), int(stride_s * FS)
    for k, x in videos.items():
        T = x.shape[-1]; dur = T / FS
        L = labels[k]
        tl = np.linspace(0, dur, len(L))
        for s in range(0, T - W + 1, S):
            seg = x[..., s:s + W]
            t0, t1 = s / FS, (s + W) / FS
            m = (tl >= t0) & (tl < t1)
            lv = L[m] if m.any() else L[[np.argmin(abs(tl - (t0 + t1) / 2))]]
            X.append(seg.reshape(-1, W))                      # (306, W)
            F.append(window_features(seg).reshape(-1))        # 6*51*6
            cv.append(lv[:, 0].mean()); ca.append(lv[:, 1].mean())
            vid.append(k); tc.append((t0 + t1) / 2)
    cv, ca = np.array(cv), np.array(ca)
    return (np.stack(X), np.stack(F), to_class3(cv), to_class3(ca), cv, ca,
            np.array(vid), np.array(tc))


def fit_predict_model(m, F, X, y, tr_all, tr, va, te, seed=0, epochs=40):
    """Train model m on training subjects, return class probabilities (n_test, 3) for te."""
    if m == "svm":
        clf = make_pipeline(StandardScaler(), PCA(n_components=min(64, F.shape[1], len(tr_all) - 1), random_state=seed),
                            SVC(kernel="rbf", C=1.0, class_weight="balanced", decision_function_shape="ovr"))
        clf.fit(F[tr_all], y[tr_all])          # no early stopping -> use all training subjects
        dec = clf.decision_function(F[te])
        if dec.ndim == 1:
            dec = np.stack([-dec, dec], 1)
        p = np.zeros((len(te), 3)); p[:, clf.classes_] = softmax(dec)
        return p
    if m == "xgb":
        clf = make_xgb(seed)
        clf.fit(F[tr], y[tr], eval_set=[(F[va], y[va])], verbose=False)
        return clf.predict_proba(F[te])
    from nets import fit_predict
    return fit_predict(m, X[tr], y[tr], X[va], y[va], X[te], 3, epochs=epochs, seed=seed)


def make_xgb(seed=0, n=600):
    return XGBClassifier(n_estimators=n, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.4,
                         tree_method="hist", early_stopping_rounds=40, eval_metric="mlogloss", num_class=3,
                         objective="multi:softprob", random_state=seed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", default="./out")
    ap.add_argument("--win", type=float, default=10.0)
    ap.add_argument("--stride", type=float, default=2.0)
    ap.add_argument("--folds", type=int, default=8)
    ap.add_argument("--models", default="svm,xgb,cnn", help="comma list; add 'transformer' if resources permit")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--video-holdout-models", default="svm,xgb",
                    help="models also evaluated with BOTH subjects and videos held out (stimulus-confound check); '' to skip")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    models = a.models.split(",")

    t = time.time()
    X, F, YV, YA, CV, CA, SUB, VID, TC = [], [], [], [], [], [], [], [], []
    norm_srcs = set()
    sids = subject_ids(a.root)
    for sid in sids:
        v, l, ns = load_subject(a.root, sid); norm_srcs.add(ns)
        x, f, yv, ya, cv, ca, vid, tc = make_windows(v, l, a.win, a.stride)
        X.append(x); F.append(f); YV.append(yv); YA.append(ya); CV.append(cv); CA.append(ca)
        VID.append(vid); TC.append(tc); SUB.append(np.full(len(yv), int(re.sub(r"\D", "", sid))))
        print(f"[load] subject {sid}: {len(yv)} windows ({len(v)} videos)", flush=True)
    X, F = np.concatenate(X), np.concatenate(F)
    Y = {"valence": np.concatenate(YV), "arousal": np.concatenate(YA)}
    SUB, VID, TC = np.concatenate(SUB), np.concatenate(VID), np.concatenate(TC)
    n_rec = int(len(set(zip(SUB.tolist(), VID.tolist()))))
    print(f"[load] {len(SUB)} windows, {len(set(SUB))} subjects, {n_rec} recordings, {time.time()-t:.0f}s")

    oof = {m: {d: np.zeros((len(SUB), 3), np.float32) for d in Y} for m in models}
    fold_scores = {m: {d: [] for d in Y} for m in models}
    gkf = GroupKFold(n_splits=min(a.folds, len(set(SUB))))
    rng = np.random.default_rng(a.seed)

    for fi, (tr_all, te) in enumerate(gkf.split(F, groups=SUB)):
        tr_subs = np.unique(SUB[tr_all]); rng.shuffle(tr_subs)
        va_subs = tr_subs[: max(1, len(tr_subs) // 8)]
        va = tr_all[np.isin(SUB[tr_all], va_subs)]
        tr = tr_all[~np.isin(SUB[tr_all], va_subs)]
        assert_subject_disjoint(SUB[tr], SUB[te], SUB[va])
        print(f"[fold {fi}] test subjects {sorted(set(SUB[te].tolist()))}", flush=True)
        for d, y in Y.items():
            for m in models:
                p = fit_predict_model(m, F, X, y, tr_all, tr, va, te, a.seed, a.epochs)
                oof[m][d][te] = p
                s = score(y[te], p.argmax(1), 3)
                fold_scores[m][d].append(s)
                print(f"   {m:12s} {d:8s} bacc {s['bacc']:.3f}  f1 {s['f1_macro']:.3f}  acc {s['acc']:.3f}  chance {s['chance_majority']:.3f}", flush=True)

    # ---- stimulus-confound check: every participant watched the same 15 videos, so a model could learn
    # "this looks like video 4" instead of "this is high arousal". Hold out people AND videos together.
    vh_models = [m for m in a.video_holdout_models.split(",") if m]
    vh_scores = {m: {d: [] for d in Y} for m in vh_models}
    vids = np.unique(VID); vrng = np.random.default_rng(a.seed + 1); vrng.shuffle(vids)
    vfolds = np.array_split(vids, 5)                                   # 5 groups of 3 videos
    for fi, (tr_all0, te0) in enumerate(gkf.split(F, groups=SUB)):
        test_vids = vfolds[fi % len(vfolds)]
        te = te0[np.isin(VID[te0], test_vids)]
        tr_all = tr_all0[~np.isin(VID[tr_all0], test_vids)]
        tr_subs = np.unique(SUB[tr_all]); rng.shuffle(tr_subs)
        va_subs = tr_subs[: max(1, len(tr_subs) // 8)]
        va = tr_all[np.isin(SUB[tr_all], va_subs)]; tr = tr_all[~np.isin(SUB[tr_all], va_subs)]
        assert_subject_disjoint(SUB[tr], SUB[te], SUB[va])
        assert not set(VID[tr_all].tolist()) & set(VID[te].tolist()), "video leak"
        for d, y in Y.items():
            for m in vh_models:
                p = fit_predict_model(m, F, X, y, tr_all, tr, va, te, a.seed, a.epochs)
                s = score(y[te], p.argmax(1), 3); vh_scores[m][d].append(s)
                print(f"[subj+video fold {fi}] {m:6s} {d:8s} bacc {s['bacc']:.3f}  f1 {s['f1_macro']:.3f}", flush=True)

    results = {
        "dataset": "REFED", "modality": "fNIRS only",
        "protocol": f"subject-independent GroupKFold ({gkf.n_splits} folds), 3-class, {a.win:g}s windows / {a.stride:g}s stride",
        "thresholds": [0.4, 0.6], "normalisation": sorted(norm_srcs),
        "n_subjects": int(len(set(SUB))), "n_recordings": n_rec, "n_windows": int(len(SUB)),
        "fs_raw_hz": round(FS_RAW, 2), "signal_types": len(SIGNALS),
        "models": {m: {d: summarise(fold_scores[m][d]) for d in Y} for m in models},
        "headline_metric": "bacc",
        "subject_video_holdout": {
            "protocol": "test people AND test videos both unseen in training (5 video groups of 3, paired with subject folds)",
            "models": {m: {d: summarise(vh_scores[m][d]) for d in Y} for m in vh_models}} if vh_models else None,
    }
    dump(results, os.path.join(a.out, "refed_results.json"))
    np.savez_compressed(os.path.join(a.out, "refed_oof.npz"), subject=SUB, video=VID, t=TC,
                        y_valence=Y["valence"], y_arousal=Y["arousal"],
                        c_valence=np.concatenate(CV), c_arousal=np.concatenate(CA),
                        **{f"p_{m}_{d}": oof[m][d] for m in models for d in Y})
    # feature cache for personalize.py (avoids re-reading 6 GB of raw data)
    np.savez(os.path.join(a.out, "refed_features.npz"), F=F.astype(np.float32), subject=SUB, video=VID,
             y_valence=Y["valence"], y_arousal=Y["arousal"])
    print("[done]", os.path.join(a.out, "refed_results.json"))


if __name__ == "__main__":
    main()
