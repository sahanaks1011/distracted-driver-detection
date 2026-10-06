"""Build the two-page write-up PDF from the real results (run after src.report).

Owners: both (Part A, Sahana: sections 1 to 4; Part B, Rachel: sections 5 to 7)
Run:   python -m src.make_writeup --repo_url https://github.com/<user>/distracted-driver-detection
Output: results/Team7_Writeup_Distracted_Driver_Detection.pdf

Read the generated PDF once and adjust any sentence in this file that you want to phrase
differently. Every number is filled in automatically from results/summary.json.
"""
import argparse
import json
from xml.sax.saxutils import escape
from datetime import date

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from .config import FIG_DIR, RESULTS_DIR, SHORT

TEAM = [("Sahana K S", "PES2UG24CS425"), ("Rachel Sara Mathew", "PES2UG24CS390")]

body = ParagraphStyle("b", fontName="Helvetica", fontSize=9.3, leading=12.2,
                      alignment=TA_JUSTIFY, spaceAfter=4)
h = ParagraphStyle("h", fontName="Helvetica-Bold", fontSize=10.5, leading=13,
                   spaceBefore=5, spaceAfter=2, textColor=colors.HexColor("#1F3B63"))
title = ParagraphStyle("t", fontName="Helvetica-Bold", fontSize=14, leading=17, alignment=TA_CENTER)
sub = ParagraphStyle("s", fontName="Helvetica", fontSize=9, leading=11.5, alignment=TA_CENTER)
cap = ParagraphStyle("c", fontName="Helvetica", fontSize=8, leading=10, alignment=TA_CENTER)


def pct(v):
    return "n/a" if v is None else f"{v:.1f}%"


def row_for(table, key):
    return next((r for r in table if r["key"] == key), {})


def build(repo_url, out):
    S = json.loads((RESULTS_DIR / "summary.json").read_text())
    T = S["table"]
    best_r = S["best"].get("random", {})
    best_d = S["best"].get("driver", {})
    nn = row_for(T, "two_layer_net")
    cnn = row_for(T, "resnet18_transfer")
    nb = row_for(T, "naive_bayes")

    story = [
        Paragraph("Machine Learning Techniques for Distracted Driver Detection", title),
        Spacer(1, 3),
        Paragraph("UE24CS352A Machine Learning Mini-Project | Section G | Team 7 | Problem 75", sub),
        Paragraph(" | ".join(f"{n} ({s})" for n, s in TEAM) + f" | {date.today():%d %B %Y}", sub),
        Paragraph(f"Repository: {escape(repo_url)}", sub),
        Spacer(1, 6),
    ]
    P = lambda t: story.append(Paragraph(t, body))
    H = lambda t: story.append(Paragraph(t, h))

    H("1. Problem Statement")
    P("Distracted driving is a leading cause of road accidents, and different distractions carry "
      "different risk. Given a single dashboard camera image of a driver, the task is to classify it "
      "into one of ten activities: safe driving or one of nine distractions (texting or phoning with "
      "either hand, operating the radio, drinking, reaching behind, hair and makeup, talking to a "
      "passenger). We reproduce the classical models of the reference CS229 report (Feng and Yue, "
      "Stanford, 2019) and extend the study in two directions: a driver-independent evaluation and a "
      "convolutional network.")

    H("2. Dataset")
    P("We use the State Farm Distracted Driver Detection dataset (Kaggle): 22,424 labelled 640x480 "
      "RGB images in ten classes c0 to c9, recorded from 26 drivers, with roughly 2,000 to 2,500 images "
      "per class, so the classes are close to balanced. Each image is resized to 64x64x3 with OpenCV and "
      "flattened into a 12,288-dimensional vector, exactly as in the paper, then standardised per feature "
      "using training-set statistics only. The CNN instead reads the original images at 224x224.")
    P("Because consecutive frames of the same driver are nearly identical, a shuffled split places almost "
      "duplicate images in both training and validation. We therefore evaluate every model on two 80/20 "
      "splits: a random stratified split (the paper's protocol) and a driver-wise split in which no driver "
      "appears in both sets.")

    H("3. Approach")
    P("Five models follow the paper. A multiclass linear SVM (hinge loss with L2 penalty) and a softmax "
      "classifier (cross-entropy with L2 penalty) are trained with mini-batch SGD. A two-layer neural network "
      "has 100 ReLU hidden units and a softmax output, trained with SGD plus momentum and per-epoch learning "
      "rate decay, using Kaiming initialisation, which the paper proposed as future work. Gaussian Naive Bayes "
      "and a Decision Tree use scikit-learn defaults. As an extension we fine-tune an ImageNet-pretrained "
      "ResNet-18, the CNN direction the paper names as future work, with colour and small affine augmentation "
      "but no horizontal flips, since flipping would swap left-hand and right-hand classes.")
    P("Hyperparameters (learning rate and regularisation strength) are chosen by grid search on a tuning split "
      "carved out of the training data with the same splitting rule; the model is then refit on the full "
      "training set and scored once on the untouched validation set. Metrics are accuracy, macro F1, "
      "per-class accuracy and the confusion matrix.")

    H("4. Implementation")
    P("The project is a Python package (src/) run in Google Colab with a T4 GPU, with results saved to a shared Google Drive folder. data.py loads, resizes and "
      "caches the images and builds both splits; models_scratch.py implements the SVM, softmax and two-layer "
      "network in NumPy only, verified with a numerical gradient check; train_scratch.py and train_sklearn.py "
      "train the classical models; train_cnn.py fine-tunes ResNet-18 in PyTorch; evaluate.py and report.py "
      "produce metrics, confusion matrices and the comparison table; predict.py and app.py provide the live "
      "demonstration. The README lists the exact commands.")

    H("5. Results")
    hdr = ["Model", "Paper val", "Random train", "Random val", "Driver train", "Driver val"]
    data = [hdr] + [[r["model"], pct(r.get("paper_val")), pct(r.get("random_train")),
                     pct(r.get("random_val")), pct(r.get("driver_train")), pct(r.get("driver_val"))]
                    for r in T]
    t = Table(data, colWidths=[4.2 * cm] + [2.45 * cm] * 5)
    t.setStyle(TableStyle([
        ("FONT", (0, 0), (-1, -1), "Helvetica", 8.3),
        ("FONT", (0, 0), (-1, 0), "Helvetica-Bold", 8.3),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#DCE6F2")),
        ("ALIGN", (1, 0), (-1, -1), "CENTER"),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#999999")),
        ("TOPPADDING", (0, 0), (-1, -1), 2), ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
    ]))
    story += [t, Paragraph("Table 1. Training and validation accuracy for both splits; paper column from the "
                           "reference report (random split).", cap), Spacer(1, 4)]
    if (FIG_DIR / "accuracy_comparison.png").exists():
        story += [Image(str(FIG_DIR / "accuracy_comparison.png"), width=15.5 * cm, height=6.97 * cm),
                  Paragraph("Figure 1. Validation accuracy per model and split.", cap), Spacer(1, 4)]

    rv = [r.get("random_val") for r in T if r.get("random_val") is not None]
    if best_r:
        txt = f"On the random split the best model is {best_r['pretty']} at {pct(100 * best_r['val_acc'])}"
        if nn.get("random_val") is not None:
            txt += (f", and our from-scratch two-layer network reaches {pct(nn['random_val'])} "
                    "against 92.2% in the paper")
        txt += ". "
        if nb.get("random_val") is not None and rv and nb["random_val"] == min(rv):
            txt += (f"Naive Bayes is the weakest model ({pct(nb['random_val'])}), as in the paper, because its "
                    "pixel independence assumption does not hold for images.")
        P(txt)
    gap = None
    if best_d:
        if nn.get("random_val") is not None and nn.get("driver_val") is not None:
            gap = nn["random_val"] - nn["driver_val"]
        conf = best_d["top_confusions"][0] if best_d["top_confusions"] else None
        worst = min(range(10), key=lambda k: best_d["per_class_acc"][k])
        txt = ""
        if gap is not None:
            word = "falls sharply" if gap >= 10 else ("falls" if gap > 0 else "does not fall")
            txt += (f"On the driver-wise split the two-layer network {word} from {pct(nn['random_val'])} "
                    f"to {pct(nn['driver_val'])}. ")
        txt += (f"The best driver-independent model is {best_d['pretty']} at {pct(100 * best_d['val_acc'])}. "
                f"Its weakest class is {SHORT[worst]} ({pct(100 * best_d['per_class_acc'][worst])})")
        if conf:
            txt += (f", and its most frequent error is predicting {conf['pred']} for {conf['true']} "
                    f"({conf['count']} images)")
        P(txt + ".")

    H("6. Conclusions")
    lines = []
    if nn.get("random_val") is not None:
        diff = nn["random_val"] - 92.24
        lines.append("We reproduced the five classical models of the reference report on the same random "
                     "split; our two-layer network is "
                     + (f"within {abs(diff):.1f} points of" if abs(diff) < 5 else
                        f"{abs(diff):.1f} points {'above' if diff > 0 else 'below'}")
                     + " the published 92.2%.")
    if gap is not None and gap >= 10:
        lines.append("The high random-split accuracy largely reflects near-duplicate frames of the same "
                     "drivers appearing in both training and validation; on unseen drivers accuracy drops "
                     "by " + f"{gap:.1f}" + " points, so the driver-wise split is the honest estimate for "
                     "real deployment.")
    elif gap is not None:
        lines.append("Accuracy on unseen drivers stays close to the random-split figure, so the models "
                     "generalise across people.")
    if cnn.get("driver_val") is not None and nn.get("driver_val") is not None:
        d2 = cnn["driver_val"] - nn["driver_val"]
        lines.append(f"Transfer learning with ResNet-18 reaches {pct(cnn['driver_val'])} on unseen drivers, "
                     + (f"{d2:.1f} points above the from-scratch two-layer network, which confirms the paper's "
                        "expectation that convolutional features suit this task." if d2 > 0 else
                        "which did not beat the from-scratch two-layer network in our runs."))
    lines.append("Remaining errors concentrate on visually similar gestures that differ only in the "
                 "object held or the exact hand position. Future work includes semi-supervised use of the "
                 "79,726 unlabelled test images and hand or object localisation before classification.")
    P(" ".join(lines))

    H("7. Individual Contributions")
    P("Part A, Sahana K S: data pipeline and both splitting protocols, NumPy implementations of the linear SVM, "
      "softmax and two-layer network with gradient checking, hyperparameter search, repository setup, "
      "write-up sections 1 to 4.")
    P("Part B, Rachel Sara Mathew: Naive Bayes and Decision Tree baselines, ResNet-18 transfer learning, evaluation "
      "module and error analysis, result aggregation and figures, demo scripts, write-up sections 5 to 7.")

    SimpleDocTemplate(str(out), pagesize=A4, leftMargin=1.6 * cm, rightMargin=1.6 * cm,
                      topMargin=1.3 * cm, bottomMargin=1.3 * cm,
                      title="Distracted Driver Detection, Team 7").build(story)
    print("Saved", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo_url", default="https://github.com/<your-username>/distracted-driver-detection")
    a = ap.parse_args()
    build(a.repo_url, RESULTS_DIR / "Team7_Writeup_Distracted_Driver_Detection.pdf")
