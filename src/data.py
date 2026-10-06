"""Dataset loading, caching, train/validation splitting and preprocessing.

Owner: Sahana K S
Run:   python -m src.data            (auto-detects the dataset on Kaggle / Colab)
       python -m src.data --data_dir /path/to/state-farm-distracted-driver-detection
"""
import argparse
import glob
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split

from .config import CACHE_DIR, FIG_DIR, IMG_SIZE, ROOT, SEED, SHORT, VAL_FRAC

CSV_NAME = "driver_imgs_list.csv"


def find_data_dir(data_dir=None):
    """Return the folder that contains driver_imgs_list.csv and imgs/train."""
    if data_dir:
        return Path(data_dir)
    roots = ["/content/data", str(ROOT / "data"), "/kaggle/input"]
    for r in roots:
        hits = glob.glob(r + "/" + CSV_NAME) + glob.glob(r + "/*/" + CSV_NAME) \
            + glob.glob(r + "/*/*/" + CSV_NAME)
        if hits:
            return Path(hits[0]).parent
    raise FileNotFoundError(
        "Could not find driver_imgs_list.csv. Pass --data_dir pointing to the "
        "unzipped State Farm dataset folder.")


def train_image_dir(data_dir):
    for sub in ("imgs/train", "train", "imgs/imgs/train"):
        p = Path(data_dir) / sub
        if (p / "c0").is_dir():
            return p
    raise FileNotFoundError(f"No imgs/train/c0 folder found under {data_dir}")


def load_index(data_dir=None):
    """DataFrame with one row per labelled image: subject, classname, img, path, label.

    Row order is fixed (sorted), so every script builds identical splits.
    """
    data_dir = find_data_dir(data_dir)
    df = pd.read_csv(Path(data_dir) / CSV_NAME)
    df = df.sort_values(["classname", "img"]).reset_index(drop=True)
    img_dir = train_image_dir(data_dir)
    df["path"] = [str(img_dir / c / f) for c, f in zip(df.classname, df.img)]
    df["label"] = df.classname.str[1].astype(int)
    return df


def read_image(path, size=IMG_SIZE):
    img = cv2.imread(str(path))                # BGR, 480x640x3
    if img is None:
        raise IOError(f"Cannot read {path}")
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def load_dataset(data_dir=None, size=IMG_SIZE, use_cache=True):
    """Return X (N, size, size, 3) uint8, y (N,), drivers (N,)."""
    cache = CACHE_DIR / f"statefarm_{size}.npz"
    if use_cache and cache.exists():
        d = np.load(cache, allow_pickle=True)
        return d["X"], d["y"], d["drivers"]
    df = load_index(data_dir)
    print(f"Reading {len(df)} images and resizing to {size}x{size} (takes a few minutes) ...")
    with ThreadPoolExecutor(max_workers=os.cpu_count() or 4) as ex:
        imgs = list(ex.map(lambda p: read_image(p, size), df.path))
    X = np.stack(imgs).astype(np.uint8)
    y = df.label.values.astype(np.int64)
    drivers = df.subject.values.astype(str)
    np.savez_compressed(cache, X=X, y=y, drivers=drivers)
    print(f"Cached to {cache}")
    return X, y, drivers


def make_split(y, drivers, mode, val_frac=VAL_FRAC, seed=SEED):
    """Indices for train / validation.

    random : shuffled stratified 80/20 split, exactly what the paper did.
    driver : no driver appears in both sets, so it measures generalisation to new people.
    """
    idx = np.arange(len(y))
    if mode == "random":
        tr, va = train_test_split(idx, test_size=val_frac, stratify=y, random_state=seed)
    elif mode == "driver":
        gss = GroupShuffleSplit(n_splits=1, test_size=val_frac, random_state=seed)
        tr, va = next(gss.split(idx, y, groups=drivers))
    else:
        raise ValueError(mode)
    return np.sort(tr), np.sort(va)


def flatten_standardize(X_train, *others):
    """Flatten images to vectors; standardise every feature with train statistics only."""
    Xt = X_train.reshape(len(X_train), -1).astype(np.float32)
    mean = Xt.mean(0)
    std = Xt.std(0) + 1e-6
    Xt -= mean
    Xt /= std
    out = [Xt]
    for X in others:
        Z = X.reshape(len(X), -1).astype(np.float32)
        Z -= mean
        Z /= std
        out.append(Z)
    return out, (mean, std)


def save_overview_figures(X, y, drivers):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 5, figsize=(12, 5.2))
    rng = np.random.default_rng(SEED)
    for c, ax in enumerate(axes.flat):
        ax.imshow(X[rng.choice(np.where(y == c)[0])])
        ax.set_title(f"c{c} {SHORT[c]}", fontsize=10)
        ax.axis("off")
    fig.suptitle("One sample per class (64x64 after preprocessing)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "samples.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 3))
    ax.bar([f"c{i}" for i in range(10)], np.bincount(y, minlength=10), color="#4C72B0")
    ax.set_ylabel("images")
    ax.set_title(f"Class distribution ({len(y)} images, {len(set(drivers))} drivers)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "class_distribution.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Read, resize and cache the dataset")
    ap.add_argument("--data_dir", default=None)
    ap.add_argument("--size", type=int, default=IMG_SIZE)
    a = ap.parse_args()
    X, y, drivers = load_dataset(a.data_dir, a.size, use_cache=False)
    print("X", X.shape, "y", y.shape, "distinct drivers", len(set(drivers)))
    for m in ("random", "driver"):
        tr, va = make_split(y, drivers, m)
        shared = len(set(drivers[tr]) & set(drivers[va]))
        print(f"{m:7s} split: train {len(tr)}  val {len(va)}  drivers in both sets: {shared}")
    save_overview_figures(X, y, drivers)
    print("Saved results/figures/samples.png and class_distribution.png")
