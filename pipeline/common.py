"""Shared helpers for the TEMPLE / WITHIN pipeline.

Everything here is deliberately boring: subject-grouped splits, honest metrics,
and a chance baseline next to every number we report.
"""
import json
import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

# REFED paper (Ning et al., NeurIPS 2025): normalised labels are split into
# low (0-0.4), medium (0.4-0.6) and high (0.6-1.0).
# Note: the public REFED code defaults to [0.3, 0.7]; we follow the paper.
REFED_THRESHOLDS = (0.4, 0.6)


def to_class3(x01, thresholds=REFED_THRESHOLDS):
    x01 = np.asarray(x01, dtype=float)
    lo, hi = thresholds
    return np.where(x01 < lo, 0, np.where(x01 < hi, 1, 2)).astype(int)


def window_features(x):
    """x: (..., T) -> (..., 6) simple temporal statistics per channel."""
    T = x.shape[-1]
    t = np.arange(T) - (T - 1) / 2
    slope = (x * t).sum(-1) / (t ** 2).sum()
    return np.stack([
        x.mean(-1), x.std(-1), slope,
        x.min(-1), x.max(-1), x[..., -1] - x[..., 0],
    ], axis=-1)


def score(y_true, y_pred, n_classes):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    counts = np.bincount(y_true, minlength=n_classes)
    return {
        "acc": float(accuracy_score(y_true, y_pred)),
        "bacc": float(balanced_accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro",
                                   labels=list(range(n_classes)), zero_division=0)),
        # Accuracy of always predicting the most common class in THIS test fold.
        "chance_majority": float(counts.max() / max(1, counts.sum())),
        # Chance for BALANCED accuracy: sklearn averages recall over the classes present in y_true,
        # so a constant predictor scores 1 / (number of classes present), not 1 / n_classes.
        "chance_bacc": float(1.0 / max(1, int((counts > 0).sum()))),
        "n": int(len(y_true)),
    }


def summarise(fold_scores):
    out = {}
    for k in fold_scores[0]:
        v = np.array([f[k] for f in fold_scores], dtype=float)
        out[k] = float(v.mean())
        out[k + "_std"] = float(v.std())
    out["folds"] = len(fold_scores)
    return out


def softmax(z, axis=-1):
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def assert_subject_disjoint(train_groups, test_groups, val_groups=()):
    tr, te, va = set(train_groups), set(test_groups), set(val_groups)
    assert not (tr & te), f"subject leak train/test: {tr & te}"
    assert not (va & te), f"subject leak val/test: {va & te}"
    assert not (tr & va), f"subject leak train/val: {tr & va}"


def dump(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=1)


def load_mat(path):
    """scipy.io.loadmat, with a fallback for MATLAB v7.3 (HDF5) files, so real REFED files load either way."""
    from scipy.io import loadmat
    try:
        return loadmat(path)
    except (NotImplementedError, ValueError):
        import h5py
        with h5py.File(path, "r") as f:
            # MATLAB stores arrays column-major; h5py sees them transposed
            return {k: np.array(f[k]).T for k in f.keys() if isinstance(f[k], h5py.Dataset)}


def mat_keys(path):
    from scipy.io import whosmat
    try:
        return [name for name, *_ in whosmat(path)]
    except (NotImplementedError, ValueError):
        import h5py
        with h5py.File(path, "r") as f:
            return list(f.keys())


def refed_label_path(root, sid):
    """REFED labels are annotations/<id>_label.mat on Hugging Face; older code used s<id>_label.mat."""
    import os
    for name in (f"{sid}_label.mat", f"s{sid}_label.mat"):
        p = os.path.join(root, "annotations", name)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"no label file for subject {sid} in {os.path.join(root, 'annotations')}")
