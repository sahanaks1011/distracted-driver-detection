"""Metrics, confusion matrices and result files shared by every training script.

Owner: Rachel Sara Mathew
"""
import json

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from .config import FIG_DIR, NUM_CLASSES, RESULTS_DIR, SHORT


def evaluate(y_true, y_pred, y_train=None, y_train_pred=None):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES)))
    per_class = (np.diag(cm) / np.maximum(cm.sum(1), 1)).tolist()
    off = cm.copy()
    np.fill_diagonal(off, 0)
    top = []
    for flat in np.argsort(off, axis=None)[::-1][:5]:
        i, j = np.unravel_index(flat, off.shape)
        if off[i, j] > 0:
            top.append({"true": SHORT[i], "pred": SHORT[j], "count": int(off[i, j])})
    out = {
        "val_acc": float(accuracy_score(y_true, y_pred)),
        "val_macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "per_class_acc": per_class,
        "confusion_matrix": cm.tolist(),
        "top_confusions": top,
    }
    if y_train is not None:
        out["train_acc"] = float(accuracy_score(y_train, y_train_pred))
    return out


def already_done(model, split, force=False):
    """True if this model/split already has a result file (lets you rerun after a disconnect)."""
    if not force and (RESULTS_DIR / f"{model}_{split}.json").exists():
        print(f"  skip {model} [{split}]: result already saved (use --force to retrain)")
        return True
    return False


def save_result(model, split, metrics, extra=None):
    rec = {"model": model, "split": split, **metrics, **(extra or {})}
    path = RESULTS_DIR / f"{model}_{split}.json"
    path.write_text(json.dumps(rec, indent=2))
    tr = f"train {rec['train_acc']:.4f}  " if "train_acc" in rec else ""
    print(f"  >> {model} [{split}]  {tr}val {rec['val_acc']:.4f}  macroF1 {rec['val_macro_f1']:.4f}")
    return rec


def plot_confusion(cm, title, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cm = np.asarray(cm, dtype=float)
    norm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(NUM_CLASSES), SHORT, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(NUM_CLASSES), SHORT, fontsize=8)
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            if norm[i, j] >= 0.01:
                ax.text(j, i, f"{norm[i, j] * 100:.0f}", ha="center", va="center", fontsize=7,
                        color="white" if norm[i, j] > 0.5 else "black")
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title, fontsize=10)
    fig.colorbar(im, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_curve(values, title, ylabel, path, labels=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 3.4))
    series = values if isinstance(values, (list, tuple)) and values and isinstance(values[0], (list, tuple)) else [values]
    for k, v in enumerate(series):
        ax.plot(v, label=None if labels is None else labels[k])
    ax.set_title(title, fontsize=10)
    ax.set_ylabel(ylabel)
    if labels:
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def figure_path(name):
    return FIG_DIR / name
