"""Datasets: a synthetic bearing-vibration generator (works now) and drop-in loaders for the
real CWRU / Paderborn bearing datasets.

All loaders return RAW windowed signals so the trigger can be applied in the input domain:
    X_train, y_train, X_test, y_test : float32 arrays of shape (N, sig_len), labels in [0..C-1]
    meta : dict with 'classes', 'fs', 'source'

REAL DATA CONVENTION (keeps you out of CWRU/Paderborn's messy native naming):
    organise .mat files into one sub-folder per class, e.g.
        data/cwru/{normal,inner,outer,ball}/*.mat
        data/paderborn/{healthy,inner,outer}/*.mat
    The loader reads the first 1-D vibration array in each file and slices it into windows.
    See ../data/README.md.
"""
import os
import glob
import numpy as np

# ----------------------------------------------------------------------------- synthetic
_DEFECT_FREQ = {0: None, 1: 0.050, 2: 0.070, 3: 0.090}  # cycles/sample, inside the feature band


def _make_signal(rng, cls, N):
    t = np.arange(N)
    sig = np.zeros(N)
    for k, a in [(1, 1.0), (2, 0.5), (3, 0.3)]:
        sig += a * np.sin(2 * np.pi * 0.010 * k * t + rng.uniform(0, 2 * np.pi))
    fd = _DEFECT_FREQ[cls]
    if fd is not None:
        amp = 1.0 + 0.2 * rng.standard_normal()
        sig += amp * np.sin(2 * np.pi * fd * t + rng.uniform(0, 2 * np.pi))
        sig += 0.4 * amp * np.sin(2 * np.pi * 2 * fd * t + rng.uniform(0, 2 * np.pi))
    sig += 0.25 * rng.standard_normal(N)
    return sig.astype(np.float32)


def _load_synthetic(sig_len, seed, n_per_class=700, n_classes=4):
    rng = np.random.default_rng(seed)
    X, y = [], []
    for c in range(n_classes):
        for _ in range(n_per_class):
            X.append(_make_signal(rng, c, sig_len)); y.append(c)
    X = np.asarray(X, np.float32); y = np.asarray(y, np.int64)
    idx = rng.permutation(len(y)); X, y = X[idx], y[idx]
    ntr = int(0.8 * len(y))
    classes = ["normal", "inner", "outer", "ball"][:n_classes]
    return X[:ntr], y[:ntr], X[ntr:], y[ntr:], {"classes": classes, "fs": None, "source": "synthetic"}

# ----------------------------------------------------------------------------- real (.mat)
def _first_1d_array(mat):
    """Pick the vibration channel. Prefer CWRU drive-end ('DE_time'), then 'FE_time', else the
    largest 1-D numeric array. Keeps the channel consistent across files."""
    cands = []  # (priority, size, array)
    for k, v in mat.items():
        if k.startswith("__"):
            continue
        arr = np.asarray(v).squeeze()
        if arr.ndim == 1 and arr.size > 1000 and np.issubdtype(arr.dtype, np.number):
            pr = 2 if "DE_time" in k else (1 if "FE_time" in k else 0)
            cands.append((pr, arr.size, arr.astype(np.float32)))
    if not cands:
        return None
    cands.sort(key=lambda t: (t[0], t[1]))   # highest priority, then largest
    return cands[-1][2]


def _window(sig, sig_len, hop):
    out = [sig[i:i + sig_len] for i in range(0, len(sig) - sig_len + 1, hop)]
    return np.asarray(out, np.float32) if out else np.empty((0, sig_len), np.float32)


def _load_folders(root, sig_len, seed, hop=None, test_frac=0.2):
    """Generic class-folder loader. Each immediate sub-folder of `root` is a class."""
    try:
        from scipy.io import loadmat
    except ImportError as e:
        raise ImportError("Real-data loading needs scipy: `conda install -n qsentry scipy`") from e
    hop = hop or sig_len  # non-overlapping by default
    classdirs = sorted(d for d in glob.glob(os.path.join(root, "*")) if os.path.isdir(d))
    if not classdirs:
        raise FileNotFoundError(f"No class sub-folders in {root}. See data/README.md for the layout.")
    rng = np.random.default_rng(seed)
    Xtr, ytr, Xte, yte, classes = [], [], [], [], []
    for ci, cd in enumerate(classdirs):
        classes.append(os.path.basename(cd))
        segs = []
        for f in sorted(glob.glob(os.path.join(cd, "*.mat"))):
            sig = _first_1d_array(loadmat(f))
            if sig is not None:
                sig = (sig - sig.mean()) / (sig.std() + 1e-8)
                segs.append(_window(sig, sig_len, hop))
        if not segs:
            raise FileNotFoundError(f"No usable .mat signals in {cd}")
        S = np.concatenate(segs, 0)
        rng.shuffle(S)
        ntr = int((1 - test_frac) * len(S))
        Xtr.append(S[:ntr]); ytr += [ci] * ntr
        Xte.append(S[ntr:]); yte += [ci] * (len(S) - ntr)
    Xtr = np.concatenate(Xtr, 0); Xte = np.concatenate(Xte, 0)
    ytr = np.asarray(ytr, np.int64); yte = np.asarray(yte, np.int64)
    return Xtr, ytr, Xte, yte, {"classes": classes, "fs": None, "source": os.path.basename(root)}


# ----------------------------------------------------------------------------- API
def load(name="synthetic", data_dir="../data", sig_len=1024, seed=0):
    if name == "synthetic":
        return _load_synthetic(sig_len, seed)
    if name in ("cwru", "paderborn", "mfpt"):
        return _load_folders(os.path.join(data_dir, name), sig_len, seed)
    raise ValueError(f"unknown dataset '{name}'")


class Normalizer:
    """Global standardization fitted on training signals; trigger is added in normalized space."""
    def fit(self, X):
        self.mu = float(X.mean()); self.sd = float(X.std()) + 1e-8; return self

    def __call__(self, X):
        return ((X - self.mu) / self.sd).astype(np.float32)
