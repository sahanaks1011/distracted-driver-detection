"""Library baselines from the paper: Gaussian Naive Bayes and Decision Tree (scikit-learn).

Owner: Rachel Sara Mathew
Run:   python -m src.train_sklearn                   (both models, both splits)
       python -m src.train_sklearn --models nb --splits random

The paper used default hyperparameters for both, so we do the same. The decision
tree on 12288 raw pixel features takes roughly 5 to 15 minutes per split on a CPU.
"""
import argparse
import time

import joblib
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

from .config import MODELS_DIR, SEED, SPLITS
from .data import flatten_standardize, load_dataset, make_split
from .evaluate import already_done, evaluate, figure_path, plot_confusion, save_result

NAMES = {"nb": "naive_bayes", "dt": "decision_tree"}


def make(kind):
    if kind == "nb":
        return GaussianNB()
    return DecisionTreeClassifier(random_state=SEED)


def run(kind, split, X, y, drivers):
    print(f"\n=== {NAMES[kind]}  |  split = {split} ===")
    tr, va = make_split(y, drivers, split)
    (Xtr, Xva), (mean, std) = flatten_standardize(X[tr], X[va])
    model = make(kind)
    t0 = time.time()
    model.fit(Xtr, y[tr])
    secs = time.time() - t0
    metrics = evaluate(y[va], model.predict(Xva), y[tr], model.predict(Xtr))
    extra = {"train_seconds": round(secs, 1), "hyperparameters": "scikit-learn defaults"}
    if kind == "dt":
        extra["tree_depth"] = int(model.get_depth())
        extra["tree_leaves"] = int(model.get_n_leaves())
    rec = save_result(NAMES[kind], split, metrics, extra)
    joblib.dump({"model": model, "mean": mean, "std": std},
                MODELS_DIR / f"{NAMES[kind]}_{split}.joblib", compress=3)
    plot_confusion(metrics["confusion_matrix"], f"{NAMES[kind]} ({split} split)",
                   figure_path(f"cm_{NAMES[kind]}_{split}.png"))
    return rec


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["nb", "dt"], choices=["nb", "dt"])
    ap.add_argument("--splits", nargs="+", default=list(SPLITS), choices=list(SPLITS))
    ap.add_argument("--data_dir", default=None)
    ap.add_argument("--force", action="store_true", help="retrain even if a result exists")
    a = ap.parse_args()
    X, y, drivers = load_dataset(a.data_dir)
    for split in a.splits:
        for kind in a.models:
            if not already_done(NAMES[kind], split, a.force):
                run(kind, split, X, y, drivers)
