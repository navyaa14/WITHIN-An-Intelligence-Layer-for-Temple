"""Does personal data help? Measures the Moments idea instead of assuming it.

For every held-out person (never in the generic model's training set):
  1. Split their units in two: a FIXED test half and a calibration pool.
     REFED unit = one video (15 per person). Tufts unit = one n-back block (16 per person;
     test = the last 8 blocks, which contain every workload level twice).
  2. k = 0: score the generic model on the test half.
  3. k = 1, 2, 4, 8: give the model k of the person's own labelled units from the pool
     (continued XGBoost training on top of the generic model), re-score the SAME test half.
     Repeated with different random picks; averaged.
  4. Reference: a model trained ONLY on the k personal units (no population data).

Headline metric: balanced accuracy (labels are imbalanced), with macro-F1 alongside.

    python personalize.py --refed-features ./out/refed_features.npz \
        --tufts-windows <size_30sec_150ts_stride_03ts dir> --partition-plan partition_plan.txt --out ./out
Run refed_train.py first: it writes refed_features.npz.
"""
import argparse, glob, os
import numpy as np
import xgboost as xgb
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score, f1_score
from scipy.stats import wilcoxon

from common import window_features, assert_subject_disjoint, dump

KS = [0, 1, 2, 4, 8]


def metrics(y, p, n):
    return balanced_accuracy_score(y, p), f1_score(y, p, average="macro", labels=list(range(n)), zero_division=0)


def generic_booster(F, y, tr, va, n, seed):
    params = dict(objective="multi:softprob", num_class=n, max_depth=4, eta=0.05, subsample=0.8,
                  colsample_bytree=0.4, tree_method="hist", eval_metric="mlogloss", seed=seed)
    dtr, dva = xgb.DMatrix(F[tr], label=y[tr]), xgb.DMatrix(F[va], label=y[va])
    bst = xgb.train(params, dtr, 600, evals=[(dva, "va")], early_stopping_rounds=40, verbose_eval=False)
    # Keep only the trees up to the best validation iteration. Both the k = 0 score and every
    # personalised model then start from exactly the same generic model.
    return bst[: bst.best_iteration + 1], params


def paired_stats(d, seed=0, n_boot=5000):
    """Paired gain across people: mean, SEM, bootstrap 95% CI, one-sided Wilcoxon p (gain > 0)."""
    d = np.asarray(d, float)
    out = {"gain": float(d.mean()), "gain_sem": float(d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else 0.0,
           "n_improved": int((d > 0).sum()), "n_worse": int((d < 0).sum()), "n_people": int(len(d))}
    if len(d) >= 2:
        r = np.random.default_rng(seed)
        boots = d[r.integers(0, len(d), (n_boot, len(d)))].mean(1)
        out["ci95"] = [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]
    if len(d) >= 6 and np.any(d != 0):
        out["p_wilcoxon"] = float(wilcoxon(d, alternative="greater", zero_method="zsplit").pvalue)
    return out


def holm(pvals):
    """Holm-Bonferroni adjusted p-values (controls false positives across all tests in the run)."""
    p = np.asarray(pvals, float); m = len(p); order = np.argsort(p); adj = np.empty(m); run = 0.0
    for i, j in enumerate(order):
        run = max(run, min(1.0, (m - i) * p[j])); adj[j] = run
    return adj


def finalize_significance(res):
    """A personalization gain is called REAL only if, after Holm correction over every test in this run:
       (a) the paired gain over the generic model is significant and its 95% CI excludes zero, AND
       (b) the personalised model itself beats per-person chance.
    (b) matters: if the generic model lands below chance by luck, a model that collapses to one class
    'gains' without learning anything. On random data this rule found nothing, as it should."""
    rows = [r for task in res.values() if isinstance(task, dict)
            for c in (task.values() if "fine_tuned" not in task else [task]) if isinstance(c, dict) and "fine_tuned" in c
            for r in c["fine_tuned"] if r["k"] > 0]
    keys = [(r, "p_wilcoxon") for r in rows if "p_wilcoxon" in r] + [(r, "p_above_chance") for r in rows if "p_above_chance" in r]
    if keys:
        for (r, k), a in zip(keys, holm([r[k] for r, k in keys])):
            r[k + "_holm"] = float(a)
    for r in rows:
        r["significant"] = bool(r.get("ci95", [0])[0] > 0 and r.get("p_wilcoxon_holm", 1) < 0.05
                                and r.get("p_above_chance_holm", 1) < 0.05)
    res["significance_rule"] = ("gain is reported as real only if Holm-corrected paired Wilcoxon p < 0.05, the 95% "
                                f"bootstrap CI excludes zero, and the personalised model beats chance (Holm p < 0.05); "
                                f"{len(keys)} tests corrected together")


def curve(F, y, SUB, UNIT, n, splits, test_units_of, seed=0, reps=3, tag=""):
    """Per held-out person: generic score (k = 0) vs. continued boosting on k of their own units.

    Every person is scored on the same fixed test half at every k, so differences are paired.
    Chance is computed PER PERSON as 1 / (classes present in their test half), because balanced
    accuracy only averages over classes that occur. A per-person shuffled-label baseline is
    computed too, as a second chance estimate that needs no assumption.
    """
    rng = np.random.default_rng(seed)
    rows = {}                                  # person -> {k: (bacc, f1)}
    own = {}                                   # person -> {k: (bacc, f1)}  personal-data-only reference
    own_skipped = {k: 0 for k in KS if k >= 2}
    chance, shuffled, skipped = {}, {}, []
    for fi, (tr, va, te) in enumerate(splits):
        assert_subject_disjoint(SUB[tr], SUB[te], SUB[va])
        gen, params = generic_booster(F, y, tr, va, n, seed)
        for s in np.unique(SUB[te]):
            idx = te[SUB[te] == s]
            test_u, pool_u = test_units_of(UNIT[idx], int(s))
            ti = idx[np.isin(UNIT[idx], test_u)]
            present = np.unique(y[ti])
            if len(present) < 2:
                skipped.append(int(s)); continue       # one label only: balanced accuracy is meaningless
            chance[int(s)] = 1.0 / len(present)
            dte = xgb.DMatrix(F[ti])
            p0 = gen.predict(dte).argmax(1)
            perm = np.random.default_rng(seed + int(s))
            shuffled[int(s)] = float(np.mean([metrics(perm.permutation(y[ti]), p0, n)[0] for _ in range(50)]))
            rows[int(s)] = {0: metrics(y[ti], p0, n)}; own[int(s)] = {}
            for k in KS[1:]:
                if k > len(pool_u):
                    continue
                b_list, f_list, ob, of = [], [], [], []
                for r in range(reps):
                    pick = rng.choice(pool_u, k, replace=False)
                    ci = idx[np.isin(UNIT[idx], pick)]
                    tuned = xgb.train({**params, "eta": 0.1, "max_depth": 3}, xgb.DMatrix(F[ci], label=y[ci]), 60,
                                      xgb_model=gen)   # continue boosting from the (trimmed) generic model
                    b, f = metrics(y[ti], tuned.predict(dte).argmax(1), n); b_list.append(b); f_list.append(f)
                    if k >= 2:
                        if len(np.unique(y[ci])) >= 2:  # reference: personal data only
                            lr = make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=2000, class_weight="balanced"))
                            lr.fit(F[ci], y[ci]); bo, fo = metrics(y[ti], lr.predict(F[ti]), n); ob.append(bo); of.append(fo)
                        else:
                            own_skipped[k] += 1
                rows[int(s)][k] = (float(np.mean(b_list)), float(np.mean(f_list)))
                if ob:
                    own[int(s)][k] = (float(np.mean(ob)), float(np.mean(of)))
        done = [p for p in rows]
        print(f"[{tag} fold {fi}] " + "  ".join(
            f"k={k}: {np.mean([rows[p][k][0] for p in done if k in rows[p]]):.3f}" for k in KS
            if any(k in rows[p] for p in done)), flush=True)

    people = sorted(rows)
    sem = lambda v: float(np.std(v, ddof=1) / np.sqrt(len(v))) if len(v) > 1 else 0.0
    out = {"k": [], "chance_bacc": float(np.mean([chance[p] for p in people])) if people else 1.0 / n,
           "chance_bacc_note": "mean over people of 1 / (classes present in their test half)",
           "shuffled_label_bacc": float(np.mean([shuffled[p] for p in people])) if people else None,
           "n_people": len(people), "skipped_single_class": skipped,
           "fine_tuned": [], "personal_only": [], "personal_only_skipped_reps": own_skipped}
    for k in KS:
        have = [p for p in people if k in rows[p]]
        if not have:
            continue
        out["k"].append(k)
        b = np.array([rows[p][k][0] for p in have]); f = np.array([rows[p][k][1] for p in have])
        row = {"k": k, "bacc": float(b.mean()), "bacc_sem": sem(b), "f1": float(f.mean()), "n_people": len(have),
               "bacc_minus_chance": float(np.mean([rows[p][k][0] - chance[p] for p in have]))}
        if k > 0:
            st = paired_stats([rows[p][k][0] - rows[p][0][0] for p in have], seed)
            row.update(gain_vs_generic=st.pop("gain"), **st)
            row["f1_gain_vs_generic"] = float(np.mean([rows[p][k][1] - rows[p][0][1] for p in have]))
            above = np.array([rows[p][k][0] - chance[p] for p in have])
            if len(above) >= 6 and np.any(above != 0):
                row["p_above_chance"] = float(wilcoxon(above, alternative="greater", zero_method="zsplit").pvalue)
        out["fine_tuned"].append(row)
    for k in own_skipped:
        have = [p for p in people if k in own[p]]
        if have:
            b = [own[p][k][0] for p in have]
            out["personal_only"].append({"k": k, "bacc": float(np.mean(b)), "bacc_sem": sem(b),
                                         "f1": float(np.mean([own[p][k][1] for p in have])), "n_people": len(have)})
    return out


def subject_splits(SUB, folds, seed):
    rng = np.random.default_rng(seed); res = []
    for tr_all, te in GroupKFold(n_splits=min(folds, len(np.unique(SUB)))).split(SUB, groups=SUB):
        subs = np.unique(SUB[tr_all]); rng.shuffle(subs); vs = subs[: max(1, len(subs) // 8)]
        res.append((tr_all[~np.isin(SUB[tr_all], vs)], tr_all[np.isin(SUB[tr_all], vs)], te))
    return res


def refed(a):
    z = np.load(a.refed_features); F, SUB, VID = z["F"], z["subject"], z["video"]
    splits = subject_splits(SUB, a.folds, a.seed)

    def halves(units, sid):                          # fixed per person: 7 test videos, 8 calibration videos
        u = np.unique(units); r = np.random.default_rng([a.seed, sid]); r.shuffle(u)   # different split per person
        return u[:len(u) // 2], u[len(u) // 2:]
    return {d: curve(F, z[f"y_{d}"], SUB, VID, 3, splits, halves, a.seed, a.reps, f"REFED {d}") for d in ("valence", "arousal")}


def tufts(a):
    from tufts_train import read_windows, subject_from_path, ELIGIBLE, parse_plan
    X, Y, S, B, H = [], [], [], [], {}
    WIN_CHUNKS = 150 // 3                                       # a 30 s window spans 50 chunk steps (3 samples each)
    diag = {"blocks_per_person": {}, "mixed_windows_dropped": 0, "boundary_windows_dropped": 0,
            "people_test_half_missing_levels": {}}
    for p in sorted(glob.glob(os.path.join(a.tufts_windows, "*.csv"))):
        sid = subject_from_path(p)
        if sid not in ELIGIBLE:
            continue
        x, y, ch, mixed = read_windows(p, a.keep_every, return_mixed=True)
        mu, sd = x.mean((0, 2), keepdims=True), x.std((0, 2), keepdims=True) + 1e-6   # label-free, own data
        # 1. Drop windows that straddle two levels (they belong to neither block).
        keep = ~mixed; diag["mixed_windows_dropped"] += int(mixed.sum())
        x, y, ch = x[keep], y[keep], ch[keep]
        # Block id = new block whenever the level changes OR the chunk sequence jumps (a gap between blocks).
        step = np.diff(ch); gap = step > a.keep_every * 1.5
        blk = np.concatenate([[0], np.cumsum((np.diff(y) != 0) | gap)])
        nb = int(blk.max() + 1); diag["blocks_per_person"][int(sid)] = nb
        if nb != 16:
            print(f"[Tufts] WARNING subject {sid}: found {nb} blocks, expected 16", flush=True)
        # 2. Guard band: drop test-half windows that start within one window length of the
        #    calibration/test boundary, so no raw sample is shared between the two halves.
        #    Only needed when windows slide continuously across blocks (level-straddling windows
        #    existed and there is no gap in the chunk sequence at the boundary); if windows are cut
        #    per block, the halves share no samples and nothing is dropped.
        h = nb // 2; H[int(sid)] = h
        first_test_chunk = ch[blk >= h].min() if (blk >= h).any() else np.inf
        last_cal_chunk = ch[blk < h].max() if (blk < h).any() else -np.inf
        continuous = mixed.any() and (first_test_chunk - last_cal_chunk) <= a.keep_every * 1.5 + WIN_CHUNKS
        guard = (blk >= h) & (ch < first_test_chunk + WIN_CHUNKS) if continuous else np.zeros(len(y), bool)
        diag["boundary_windows_dropped"] += int(guard.sum())
        x, y, ch, blk = x[~guard], y[~guard], ch[~guard], blk[~guard]
        missing = sorted(set(range(4)) - set(y[blk >= h].tolist()))
        if missing:
            diag["people_test_half_missing_levels"][int(sid)] = missing
        X.append((x - mu) / sd); Y.append(y); S.append(np.full(len(y), sid)); B.append(blk)
    print(f"[Tufts] dropped {diag['mixed_windows_dropped']} level-straddling and "
          f"{diag['boundary_windows_dropped']} boundary windows; "
          f"{len(diag['people_test_half_missing_levels'])} people lack some level in their test half", flush=True)
    X, Y, S, B = map(np.concatenate, (X, Y, S, B)); F = window_features(X).reshape(len(X), -1)
    if a.partition_plan:
        subs = set(S.tolist()); splits = []
        for tr_s, va_s, te_s in parse_plan(a.partition_plan):
            te_s = [s for s in te_s if s in subs]
            if te_s:
                tr = np.where(np.isin(S, tr_s))[0]; va = np.where(np.isin(S, va_s))[0]
                if len(va) == 0 or len(tr) == 0:
                    pool = np.array(sorted(subs - set(te_s))); k = max(1, len(pool) // 4)
                    tr = np.where(np.isin(S, pool[k:]))[0]; va = np.where(np.isin(S, pool[:k]))[0]
                splits.append((tr, va, np.where(np.isin(S, te_s))[0]))
    else:
        splits = subject_splits(S, 17, a.seed)

    def halves(units, sid):                          # first 8 blocks = calibration pool, last 8 = test
        u = np.unique(units); h = H[sid]                     # split point fixed before any guard-band drop
        return u[u >= h], u[u < h]
    out = curve(F, Y, S, B, 4, splits, halves, a.seed, a.reps, "Tufts"); out["data_checks"] = diag
    return {"load": out}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refed-features"); ap.add_argument("--tufts-windows"); ap.add_argument("--partition-plan")
    ap.add_argument("--keep-every", type=int, default=5); ap.add_argument("--folds", type=int, default=8)
    ap.add_argument("--reps", type=int, default=3); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="./out")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    res = {"ks": KS, "metric": "balanced accuracy (macro-F1 alongside)",
           "design": "per held-out person: fixed test half; k own labelled units added by continued boosting on the generic model"}
    if a.refed_features:
        res["refed"] = refed(a)
    if a.tufts_windows:
        res["tufts"] = tufts(a)
    finalize_significance(res)
    dump(res, os.path.join(a.out, "personalization_results.json")); print("[done]")


if __name__ == "__main__":
    main()
