"""Collect every results/*.json into one comparison table and charts.

Owner: Rachel Sara Mathew
Run:   python -m src.report
Writes results/summary.csv, results/summary.json, results/summary.md and
results/figures/accuracy_comparison.png, results/figures/per_class_best.png
"""
import json

import numpy as np
import pandas as pd

from .config import FIG_DIR, RESULTS_DIR, SHORT

ORDER = ["naive_bayes", "decision_tree", "linear_svm", "softmax", "two_layer_net",
         "resnet18_transfer"]
PRETTY = {"naive_bayes": "Naive Bayes", "decision_tree": "Decision Tree",
          "linear_svm": "Linear SVM", "softmax": "Softmax",
          "two_layer_net": "Two-layer Net", "resnet18_transfer": "ResNet-18 (extension)"}
# Validation accuracy reported in the original CS229 paper (random 80/20 split)
PAPER = {"naive_bayes": 54.99, "decision_tree": 84.73, "linear_svm": 71.39,
         "softmax": 82.31, "two_layer_net": 92.24}


def load_all():
    recs = [json.loads(p.read_text()) for p in sorted(RESULTS_DIR.glob("*_*.json"))
            if p.name not in ("summary.json",)]
    return [r for r in recs if "val_acc" in r]


def main():
    recs = load_all()
    if not recs:
        raise SystemExit("No results found. Run the training scripts first.")
    rows = {}
    for r in recs:
        row = rows.setdefault(r["model"], {"model": PRETTY.get(r["model"], r["model"]),
                                           "key": r["model"]})
        s = r["split"]
        row[f"{s}_train"] = round(100 * r["train_acc"], 2) if "train_acc" in r else None
        row[f"{s}_val"] = round(100 * r["val_acc"], 2)
        row[f"{s}_f1"] = round(100 * r["val_macro_f1"], 2)
    for k, v in PAPER.items():
        if k in rows:
            rows[k]["paper_val"] = v
    df = pd.DataFrame(sorted(rows.values(), key=lambda r: ORDER.index(r["key"]) if r["key"] in ORDER else 99))
    cols = ["model", "paper_val", "random_train", "random_val", "random_f1",
            "driver_train", "driver_val", "driver_f1"]
    df = df.reindex(columns=["key"] + cols)
    df[cols].to_csv(RESULTS_DIR / "summary.csv", index=False)

    best = {}
    for s in ("random", "driver"):
        cand = [r for r in recs if r["split"] == s]
        if cand:
            b = max(cand, key=lambda r: r["val_acc"])
            best[s] = {"model": b["model"], "pretty": PRETTY.get(b["model"], b["model"]),
                       "val_acc": b["val_acc"], "per_class_acc": b["per_class_acc"],
                       "top_confusions": b["top_confusions"]}
    summary = {"table": df.replace({np.nan: None}).to_dict(orient="records"), "best": best,
               "nn_per_class": {r["split"]: r["per_class_acc"] for r in recs
                                if r["model"] == "two_layer_net"}}
    (RESULTS_DIR / "summary.json").write_text(json.dumps(summary, indent=2))

    md = ["| Model | Paper val % | Random train % | Random val % | Driver train % | Driver val % |",
          "|---|---|---|---|---|---|"]
    fmt = lambda v: "-" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.2f}"
    for r in summary["table"]:
        md.append(f"| {r['model']} | {fmt(r.get('paper_val'))} | {fmt(r.get('random_train'))} | "
                  f"{fmt(r.get('random_val'))} | {fmt(r.get('driver_train'))} | {fmt(r.get('driver_val'))} |")
    (RESULTS_DIR / "summary.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 3.6))
    x = np.arange(len(df))
    w = 0.27
    series = [("paper_val", "Paper (random split)", "#BBBBBB"),
              ("random_val", "Ours, random split", "#4C72B0"),
              ("driver_val", "Ours, driver split", "#DD8452")]
    for k, (col, lab, colr) in enumerate(series):
        vals = pd.to_numeric(df[col], errors="coerce").fillna(0).values
        bars = ax.bar(x + (k - 1) * w, vals, w, label=lab, color=colr)
        for b_, v in zip(bars, vals):
            if v > 0:
                ax.text(b_.get_x() + b_.get_width() / 2, v + 1, f"{v:.0f}", ha="center", fontsize=7)
    ax.set_xticks(x, df["model"], fontsize=8)
    ax.set_ylabel("validation accuracy (%)")
    ax.set_ylim(0, 108)
    ax.legend(fontsize=8, loc="upper left")
    ax.set_title("Validation accuracy by model and split")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "accuracy_comparison.png", dpi=150)
    plt.close(fig)

    if best:
        fig, ax = plt.subplots(figsize=(8, 3.2))
        for k, (s, colr) in enumerate([("random", "#4C72B0"), ("driver", "#DD8452")]):
            if s in best:
                ax.bar(np.arange(10) + (k - 0.5) * 0.4, 100 * np.array(best[s]["per_class_acc"]),
                       0.4, color=colr, label=f"{best[s]['pretty']} ({s} split)")
        ax.set_xticks(range(10), SHORT, fontsize=8)
        ax.set_ylabel("accuracy (%)")
        ax.set_ylim(0, 105)
        ax.legend(fontsize=8, loc="lower left")
        ax.set_title("Per-class validation accuracy of the best model")
        fig.tight_layout()
        fig.savefig(FIG_DIR / "per_class_best.png", dpi=150)
        plt.close(fig)
    print("\nWrote results/summary.csv, summary.json, summary.md and comparison figures")


if __name__ == "__main__":
    main()
