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


def _load_folders(root, sig_len, seed, hop=None, test_frac=0.2, split="window", keep_order=False):
    """Generic class-folder loader. Each immediate sub-folder of `root` is a class.

    `split` controls how windows are divided into train and test:
      "window"     shuffle all of a class's windows and split them (the default used throughout the
                   paper). Windows never overlap (hop = sig_len), but windows cut from the SAME
                   recording -- and therefore the same bearing, motor load and session -- can land on
                   both sides of the split.
      "recording"  hold out whole source files: for each class a subset of its .mat recordings is
                   assigned entirely to the test set, so no test window shares a recording with any
                   training window. With CWRU's 3-4 files per class this is leave-one-load-out, the
                   stricter protocol a reviewer asked for. Which recording is held out is chosen by
                   `seed`, so seeds also vary the held-out operating condition.

    `keep_order=True` (recording split only) returns the test windows in their original time order,
    recording by recording, with `meta["te_rec"]` giving each window's recording. Used to evaluate
    alarm aggregation over a continuous stream, where neighbouring windows are correlated. The
    training split and the choice of held-out recordings are identical to `keep_order=False`.
    """
    try:
        from scipy.io import loadmat
    except ImportError as e:
        raise ImportError("Real-data loading needs scipy: `conda install -n qsentry scipy`") from e
    if split not in ("window", "recording"):
        raise ValueError(f"unknown split '{split}' (choose 'window' or 'recording')")
    hop = hop or sig_len  # non-overlapping by default
    classdirs = sorted(d for d in glob.glob(os.path.join(root, "*")) if os.path.isdir(d))
    if not classdirs:
        raise FileNotFoundError(f"No class sub-folders in {root}. See data/README.md for the layout.")
    rng = np.random.default_rng(seed)
    Xtr, ytr, Xte, yte, classes, heldout = [], [], [], [], [], []
    te_rec, heldout_order = [], []
    for ci, cd in enumerate(classdirs):
        classes.append(os.path.basename(cd))
        segs, names = [], []
        for f in sorted(glob.glob(os.path.join(cd, "*.mat"))):
            sig = _first_1d_array(loadmat(f))
            if sig is not None:
                sig = (sig - sig.mean()) / (sig.std() + 1e-8)
                W = _window(sig, sig_len, hop)
                if len(W):
                    segs.append(W); names.append(os.path.basename(f))
        if not segs:
            raise FileNotFoundError(f"No usable .mat signals in {cd}")
        if split == "window":
            S = np.concatenate(segs, 0)
            rng.shuffle(S)
            ntr = int((1 - test_frac) * len(S))
            Xtr.append(S[:ntr]); ytr += [ci] * ntr
            Xte.append(S[ntr:]); yte += [ci] * (len(S) - ntr)
        else:  # recording-level: whole files go to one side of the split
            if len(segs) < 2:
                raise ValueError(f"{cd} has one usable recording; a recording-level split needs >=2")
            n_te = max(1, int(round(test_frac * len(segs))))
            te_idx = set(rng.choice(len(segs), n_te, replace=False).tolist())
            heldout += [f"{classes[-1]}/{names[i]}" for i in sorted(te_idx)]
            for i, W in enumerate(segs):
                if i in te_idx:
                    Xte.append(W); yte += [ci] * len(W)
                    te_rec += [len(heldout_order)] * len(W); heldout_order.append(f"{classes[-1]}/{names[i]}")
                else:
                    Xtr.append(W); ytr += [ci] * len(W)
    Xtr = np.concatenate(Xtr, 0); Xte = np.concatenate(Xte, 0)
    ytr = np.asarray(ytr, np.int64); yte = np.asarray(yte, np.int64)
    if split == "recording":  # keep class order but break the file-contiguous ordering
        itr = rng.permutation(len(ytr)); ite = rng.permutation(len(yte))
        Xtr, ytr = Xtr[itr], ytr[itr]
        if not keep_order:   # ite is always drawn so the rng stream (and every result) is unchanged
            Xte, yte = Xte[ite], yte[ite]
    meta = {"classes": classes, "fs": None, "source": os.path.basename(root), "split": split}
    if heldout:
        meta["heldout_recordings"] = heldout
    if split == "recording" and keep_order:
        # test windows stay in recording time order; te_rec[i] indexes meta["te_recordings"]
        meta["te_rec"] = np.asarray(te_rec, np.int64); meta["te_recordings"] = heldout_order
    return Xtr, ytr, Xte, yte, meta


# ----------------------------------------------------------------------------- image domain (WP1)
def _make_synthetic_image(rng, cls, size, n_classes):
    """Fallback image with a class-dependent oriented grating + blob, so the pipeline runs without a
    real download. Enough structure for a small CNN to separate classes (a stand-in until real MNIST
    is dropped in via `mnist.npz`)."""
    yy, xx = np.mgrid[0:size, 0:size] / size
    theta = np.pi * cls / max(1, n_classes)
    grating = 0.5 + 0.5 * np.sin(2 * np.pi * 4 * (xx * np.cos(theta) + yy * np.sin(theta)))
    cx, cy = rng.uniform(0.3, 0.7, 2)
    blob = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * 0.02))
    img = 0.6 * grating + 0.5 * blob + 0.05 * rng.standard_normal((size, size))
    return np.clip(img, 0, 1).astype(np.float32)


def _load_synthetic_image(size, seed, n_per_class=600, n_classes=10):
    rng = np.random.default_rng(seed)
    X = [_make_synthetic_image(rng, c, size, n_classes) for c in range(n_classes) for _ in range(n_per_class)]
    y = [c for c in range(n_classes) for _ in range(n_per_class)]
    X = np.asarray(X, np.float32); y = np.asarray(y, np.int64)
    idx = rng.permutation(len(y)); X, y = X[idx], y[idx]
    ntr = int(0.85 * len(y))
    return (X[:ntr], y[:ntr], X[ntr:], y[ntr:],
            {"classes": [str(c) for c in range(n_classes)], "fs": None, "source": "synthetic_image"})


def _resize_nn(X, size):
    """Nearest-neighbour resize of (N,H,W) images to (N,size,size) -- no scipy/PIL dependency."""
    n, h, w = X.shape
    ri = (np.arange(size) * h / size).astype(int)
    ci = (np.arange(size) * w / size).astype(int)
    return X[:, ri][:, :, ci]


def load_image(name="mnist", data_dir="../data", img_size=16, seed=0):
    """Load a low-res image dataset as (Xtr,ytr,Xte,yte,meta) with X of shape (N,img_size,img_size),
    float32 in [0,1]. Tries, in order: torchvision (if installed), a local `<data_dir>/<name>.npz`
    (Keras layout: x_train,y_train,x_test,y_test), else a synthetic-image fallback so the scaffold
    runs today. Drop a real `mnist.npz` into data/ to get the real bit-width-threshold result."""
    npz = os.path.join(data_dir, f"{name}.npz")
    try:  # 1) torchvision, if available
        import torchvision  # noqa: F401
        from torchvision import datasets
        root = os.path.join(data_dir, "torchvision")
        ds_tr = datasets.MNIST(root, train=True, download=True)
        ds_te = datasets.MNIST(root, train=False, download=True)
        Xtr = ds_tr.data.numpy().astype(np.float32) / 255.0; ytr = ds_tr.targets.numpy().astype(np.int64)
        Xte = ds_te.data.numpy().astype(np.float32) / 255.0; yte = ds_te.targets.numpy().astype(np.int64)
        src = f"{name}(torchvision)"
    except Exception:
        if os.path.exists(npz):  # 2) local Keras-style npz
            d = np.load(npz)
            Xtr = d["x_train"].astype(np.float32) / 255.0; ytr = d["y_train"].astype(np.int64)
            Xte = d["x_test"].astype(np.float32) / 255.0;  yte = d["y_test"].astype(np.int64)
            src = f"{name}(npz)"
        else:  # 3) synthetic fallback
            return _load_synthetic_image(img_size, seed)
    Xtr, Xte = _resize_nn(Xtr, img_size), _resize_nn(Xte, img_size)
    classes = [str(c) for c in range(int(max(ytr.max(), yte.max())) + 1)]
    return Xtr, ytr, Xte, yte, {"classes": classes, "fs": None, "source": src}


# ----------------------------------------------------------------------------- API
def load(name="synthetic", data_dir="../data", sig_len=1024, seed=0, split="window", keep_order=False):
    """Load a dataset as (Xtr,ytr,Xte,yte,meta). `split` selects the train/test protocol for the
    real .mat datasets: "window" (default) or the stricter "recording" (see `_load_folders`)."""
    if name == "synthetic":
        return _load_synthetic(sig_len, seed)
    if name in ("cwru", "paderborn", "mfpt"):
        return _load_folders(os.path.join(data_dir, name), sig_len, seed, split=split,
                             keep_order=keep_order)
    raise ValueError(f"unknown dataset '{name}'")


def calib_indices(n_train, size=512, seed=0, draw="representative"):
    """Indices of the windows that form the calibration set -- the small clean sample the deployer
    already holds to quantize the model, and which the monitor reuses to fit its per-class statistics.

      "ordered"         the first `size` training windows. This is what the original submission used.
                        The loaders emit windows class by class, so on CWRU and MFPT this draw covers
                        only the first one or two classes: the monitor then has per-class statistics
                        for those classes alone and falls back to the pooled statistics elsewhere,
                        while the pooled ("global") statistic is fitted on a near-single-class sample
                        and so behaves like a class-conditional one. Kept so that the effect of the
                        calibration draw can be measured directly.
      "representative"  a uniform random draw over the training split, which covers every class in
                        proportion. This is what a deployer's own held-out calibration sample looks
                        like, and it is the protocol used for the reported results.
    """
    if draw == "ordered":
        return np.arange(min(size, n_train))
    if draw == "representative":
        return np.random.default_rng(seed).permutation(n_train)[:size]
    raise ValueError(f"unknown calibration draw '{draw}' (choose 'ordered' or 'representative')")


class Normalizer:
    """Global standardization fitted on training signals; trigger is added in normalized space."""
    def fit(self, X):
        self.mu = float(X.mean()); self.sd = float(X.std()) + 1e-8; return self

    def __call__(self, X):
        return ((X - self.mu) / self.sd).astype(np.float32)
