"""Train the three from-scratch models: Linear SVM, Softmax, Two-layer Net.

Owner: Sahana K S
Run:   python -m src.train_scratch                    (both splits, all three models)
       python -m src.train_scratch --models nn --splits driver
       python -m src.train_scratch --quick            (fewer iterations, no grid search)

Protocol (same for every model, both splits):
  1. split data 80/20 into train / validation (random or driver-wise)
  2. carve a tuning split out of TRAIN (same rule) and grid-search hyperparameters on it
  3. refit on the full TRAIN set with the best hyperparameters
  4. report accuracy on the untouched VALIDATION set
"""
import argparse
import itertools
import time

import numpy as np

from .config import MODELS_DIR, SEED, SPLITS
from .data import flatten_standardize, load_dataset, make_split
from .evaluate import already_done, evaluate, figure_path, plot_confusion, plot_curve, save_result
from .models_scratch import LinearSVM, Softmax, TwoLayerNet

GRIDS = {
    "svm":     {"lr": [3e-6, 1e-5, 3e-5, 1e-4], "reg": [1e-2, 1e-1, 1.0]},
    "softmax": {"lr": [3e-6, 1e-5, 3e-5, 1e-4], "reg": [1e-2, 1e-1, 1.0]},
    "nn":      {"lr": [1e-4, 3e-4, 1e-3], "reg": [1e-4, 1e-3, 1e-2]},
}
QUICK = {"svm": {"lr": 1e-5, "reg": 1e-1}, "softmax": {"lr": 1e-5, "reg": 1e-1},
         "nn": {"lr": 3e-4, "reg": 1e-2}}
NAMES = {"svm": "linear_svm", "softmax": "softmax", "nn": "two_layer_net"}


def build_and_train(kind, Xtr, ytr, hp, iters, Xva=None, yva=None, verbose=False):
    if kind == "nn":
        model = TwoLayerNet(Xtr.shape[1], hidden=100, seed=SEED)
        hist = model.train(Xtr, ytr, Xva, yva, lr=hp["lr"], reg=hp["reg"],
                           num_iters=iters, verbose=verbose)
    else:
        model = (LinearSVM if kind == "svm" else Softmax)(seed=SEED)
        hist = {"loss": model.train(Xtr, ytr, lr=hp["lr"], reg=hp["reg"],
                                    num_iters=iters, verbose=verbose)}
    return model, hist


def run(kind, split, X, y, drivers, quick):
    print(f"\n=== {NAMES[kind]}  |  split = {split} ===")
    tr, va = make_split(y, drivers, split)
    iters_tune = 600 if quick else (1500 if kind == "nn" else 1500)
    iters_final = 1000 if quick else (3000 if kind == "nn" else 2000)

    # ---- hyperparameter search on a tuning split carved out of TRAIN
    tuning_log = []
    if quick:
        best = QUICK[kind]
    else:
        sub_tr, sub_va = make_split(y[tr], drivers[tr], split, val_frac=0.15)
        (A, B), _ = flatten_standardize(X[tr][sub_tr], X[tr][sub_va])
        yA, yB = y[tr][sub_tr], y[tr][sub_va]
        best, best_acc = None, -1
        for lr, reg in itertools.product(GRIDS[kind]["lr"], GRIDS[kind]["reg"]):
            m, _ = build_and_train(kind, A, yA, {"lr": lr, "reg": reg}, iters_tune)
            acc = float((m.predict(B) == yB).mean())
            tuning_log.append({"lr": lr, "reg": reg, "tune_acc": acc})
            print(f"  lr={lr:.0e} reg={reg:.0e}  tune acc {acc:.4f}")
            if acc > best_acc:
                best, best_acc = {"lr": lr, "reg": reg}, acc
        del A, B
    print(f"  best hyperparameters: {best}")

    # ---- refit on full train, evaluate on validation
    (Xtr, Xva), (mean, std) = flatten_standardize(X[tr], X[va])
    t0 = time.time()
    model, hist = build_and_train(kind, Xtr, y[tr], best, iters_final, Xva, y[va], verbose=True)
    secs = time.time() - t0
    metrics = evaluate(y[va], model.predict(Xva), y[tr], model.predict(Xtr))
    rec = save_result(NAMES[kind], split, metrics,
                      {"hyperparameters": best, "tuning": tuning_log,
                       "train_seconds": round(secs, 1), "iterations": iters_final})

    model.save(MODELS_DIR / f"{NAMES[kind]}_{split}.npz", mean, std)
    plot_confusion(metrics["confusion_matrix"], f"{NAMES[kind]} ({split} split)",
                   figure_path(f"cm_{NAMES[kind]}_{split}.png"))
    loss = np.convolve(hist["loss"], np.ones(25) / 25, mode="valid").tolist()
    plot_curve(loss, f"{NAMES[kind]} training loss ({split} split)", "loss",
               figure_path(f"loss_{NAMES[kind]}_{split}.png"))
    if kind == "nn":
        plot_curve([hist["train_acc"], hist["val_acc"]],
                   f"Two-layer net accuracy per epoch ({split} split)", "accuracy",
                   figure_path(f"acc_two_layer_net_{split}.png"), labels=["train", "val"])
    return rec


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["svm", "softmax", "nn"],
                    choices=["svm", "softmax", "nn"])
    ap.add_argument("--splits", nargs="+", default=list(SPLITS), choices=list(SPLITS))
    ap.add_argument("--data_dir", default=None)
    ap.add_argument("--force", action="store_true", help="retrain even if a result exists")
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()

    X, y, drivers = load_dataset(a.data_dir)
    print("Loaded", X.shape)
    for split in a.splits:
        for kind in a.models:
            if not already_done(NAMES[kind], split, a.force):
                run(kind, split, X, y, drivers, a.quick)
