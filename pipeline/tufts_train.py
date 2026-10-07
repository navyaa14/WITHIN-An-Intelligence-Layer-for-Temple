"""Tufts fNIRS2MW: 4-level n-back workload from 30-second forehead fNIRS windows.

Data:  https://tufts-hci-lab.github.io/code_and_datasets/fNIRS2MW.html  (CC-BY-4.0)
Use the official pre-processed sliding windows:
    .../band_pass_filtered/slide_window_data/size_30sec_150ts_stride_03ts/*.csv
Columns: chunk, label (0-3), AB_I_O, AB_PHI_O, AB_I_DO, AB_PHI_DO, CD_I_O, CD_PHI_O, CD_I_DO, CD_PHI_DO

Protocol:
  * 68 eligible subjects (from the official partition_plan.txt).
  * If --partition-plan is given, the official generic-model TestBuckets are used
    (test / train / val subjects per bucket). Otherwise GroupKFold over subjects.
  * Windows overlap heavily (0.6 s stride); --keep-every thins them for speed.
    Thinning never mixes subjects across splits.

    python tufts_train.py --windows <dir of csvs> --partition-plan partition_plan.txt --out ./out
"""
import argparse, glob, os, re, ast
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from xgboost import XGBClassifier

from common import window_features, score, summarise, assert_subject_disjoint, dump

COLS = ["AB_I_O", "AB_PHI_O", "AB_I_DO", "AB_PHI_DO", "CD_I_O", "CD_PHI_O", "CD_I_DO", "CD_PHI_DO"]
ELIGIBLE = [1, 13, 14, 15, 20, 21, 22, 23, 24, 25, 27, 28, 29, 31, 32, 34, 35, 36, 37, 38, 40, 42, 43, 44, 45,
            46, 47, 48, 49, 5, 51, 52, 54, 55, 56, 57, 58, 60, 61, 62, 63, 64, 65, 68, 69, 7, 70, 71, 72, 73, 74,
            75, 76, 78, 79, 80, 81, 82, 83, 84, 85, 86, 91, 92, 93, 94, 95, 97]
LEVELS = ["LOW", "MODERATE", "HIGH", "VERY HIGH"]


def subject_from_path(p):
    m = re.findall(r"\d+", os.path.basename(p))
    return int(m[0]) if m else None


def read_windows(path, keep_every, return_mixed=False):
    df = pd.read_csv(path)
    df = df[df.chunk % keep_every == 0]
    # A window whose rows carry more than one label straddles a block boundary.
    mixed = df.groupby("chunk").label.nunique() > 1
    chunks = df.chunk.values
    order = np.unique(chunks)
    T = int((chunks == order[0]).sum())
    df = df.sort_values(["chunk"], kind="stable")
    x = df[COLS].values.astype(np.float32).reshape(len(order), T, len(COLS)).transpose(0, 2, 1)  # (N, 8, 150)
    y = df.groupby("chunk").label.first().loc[order].values.astype(int)
    if return_mixed:
        return x, y, order, mixed.loc[order].values
    return x, y, order


def parse_plan(path):
    txt = open(path).read()
    # Only the main generic-model section ("64 generic pool size"): 48 train / 16 val / 4 test.
    txt = txt.split("64 generic pool size", 1)[1].split("generic pool size", 1)[0]
    buckets = []
    for b in re.split(r"TestBucket\d+:", txt)[1:]:
        get = lambda k: ast.literal_eval(re.search(k + r"\s*=\s*(\[[^\]]*\])", b).group(1))
        try:
            buckets.append((get("train_subjects"), get("val_subjects"), get("test_subjects")))
        except AttributeError:
            break
    return buckets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", required=True)
    ap.add_argument("--partition-plan", default=None)
    ap.add_argument("--out", default="./out")
    ap.add_argument("--keep-every", type=int, default=5, help="keep every k-th 0.6 s-stride window")
    ap.add_argument("--models", default="xgb,cnn")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--folds", type=int, default=17)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    models = a.models.split(",")

    X, Y, S, C = [], [], [], []
    for p in sorted(glob.glob(os.path.join(a.windows, "*.csv"))):
        sid = subject_from_path(p)
        if sid not in ELIGIBLE:
            continue
        x, y, ch, mixed = read_windows(p, a.keep_every, return_mixed=True)
        n_mixed = int(mixed.sum()); x, y, ch = x[~mixed], y[~mixed], ch[~mixed]   # drop windows spanning two levels
        # label-free per-subject normalisation (uses no labels, no other subjects)
        mu, sd = x.mean((0, 2), keepdims=True), x.std((0, 2), keepdims=True) + 1e-6
        X.append((x - mu) / sd); Y.append(y); S.append(np.full(len(y), sid)); C.append(ch)
        print(f"[load] subject {sid}: {len(y)} windows ({n_mixed} level-straddling windows dropped)", flush=True)
    X, Y, S, C = map(np.concatenate, (X, Y, S, C))
    F = window_features(X).reshape(len(X), -1)
    subs = sorted(set(S.tolist()))
    print(f"[load] {len(Y)} windows from {len(subs)} subjects")

    if a.partition_plan:
        splits = []
        for tr_s, va_s, te_s in parse_plan(a.partition_plan):
            te_s = [s for s in te_s if s in subs]
            if not te_s:
                continue
            splits.append((np.where(np.isin(S, tr_s))[0], np.where(np.isin(S, va_s))[0], np.where(np.isin(S, te_s))[0]))
        protocol = f"official Tufts generic-model TestBuckets ({len(splits)} buckets), 4-class"
    else:
        rng = np.random.default_rng(a.seed); splits = []
        for tr_all, te in GroupKFold(n_splits=min(a.folds, len(subs))).split(F, groups=S):
            tr_subs = np.unique(S[tr_all]); rng.shuffle(tr_subs)
            va_subs = tr_subs[: max(1, len(tr_subs) // 4)]
            splits.append((tr_all[~np.isin(S[tr_all], va_subs)], tr_all[np.isin(S[tr_all], va_subs)], te))
        protocol = f"subject-independent GroupKFold ({len(splits)} folds), 4-class"

    oof = {m: np.full((len(Y), 4), np.nan, np.float32) for m in models}
    fs = {m: [] for m in models}
    for fi, (tr, va, te) in enumerate(splits):
        if len(va) == 0 or len(tr) == 0:                       # subset runs: carve val from train subjects
            pool = np.unique(S[np.concatenate([tr, va])]) if len(tr) + len(va) else np.array(sorted(set(subs) - set(S[te])))
            k = max(1, len(pool) // 4)
            tr = np.where(np.isin(S, pool[k:]))[0]; va = np.where(np.isin(S, pool[:k]))[0]
        assert_subject_disjoint(S[tr], S[te], S[va])
        for m in models:
            if m == "xgb":
                clf = XGBClassifier(n_estimators=800, max_depth=4, learning_rate=0.05, subsample=0.8,
                                    colsample_bytree=0.6, tree_method="hist", early_stopping_rounds=50,
                                    eval_metric="mlogloss", random_state=a.seed)
                clf.fit(F[tr], Y[tr], eval_set=[(F[va], Y[va])], verbose=False)
                p = clf.predict_proba(F[te])
            else:
                from nets import fit_predict
                p = fit_predict(m, X[tr], Y[tr], X[va], Y[va], X[te], 4, epochs=a.epochs, seed=a.seed)
            oof[m][te] = p
            s = score(Y[te], p.argmax(1), 4); fs[m].append(s)
            print(f"[split {fi}] {m:12s} acc {s['acc']:.3f} f1 {s['f1_macro']:.3f} chance {s['chance_majority']:.3f}", flush=True)

    dump({"dataset": "Tufts fNIRS2MW", "protocol": protocol, "window_s": 30,
          "n_subjects": len(subs), "n_windows": int(len(Y)), "fs_hz": 5.2, "levels": 4,
          "ui_levels": LEVELS,
          "models": {m: summarise(fs[m]) for m in models}}, os.path.join(a.out, "tufts_results.json"))
    np.savez_compressed(os.path.join(a.out, "tufts_oof.npz"), subject=S, chunk=C, y=Y,
                        **{f"p_{m}": oof[m] for m in models})
    print("[done]")


if __name__ == "__main__":
    main()
