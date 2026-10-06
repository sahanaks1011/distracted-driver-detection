"""Extension beyond the paper: CNN transfer learning (ResNet-18 pretrained on ImageNet).

The paper's conclusion names CNNs such as ResNet / VGG as future work; we implement it.
Owner: Rachel Sara Mathew
Run (needs a GPU: in Colab, Runtime > Change runtime type > T4 GPU):
       python -m src.train_cnn                       (both splits, 4 epochs each)
       python -m src.train_cnn --splits driver --epochs 5

Notes for the viva:
  * No horizontal flipping: it would turn "texting right" into "texting left".
  * Same train / validation indices as every other model (same make_split call).
  * Fixed number of epochs; the validation set is only used for reporting.
"""
import argparse
import time

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

from .config import MODELS_DIR, NUM_CLASSES, SEED, SPLITS
from .data import load_index, make_split
from .evaluate import already_done, evaluate, figure_path, plot_confusion, plot_curve, save_result

MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
IMG = 224

train_tf = transforms.Compose([
    transforms.Resize((IMG, IMG)),
    transforms.RandomAffine(degrees=8, translate=(0.05, 0.05), scale=(0.95, 1.05)),
    transforms.ColorJitter(0.3, 0.3, 0.2),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])
eval_tf = transforms.Compose([
    transforms.Resize((IMG, IMG)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])


class Frames(Dataset):
    def __init__(self, paths, labels, tf):
        self.paths, self.labels, self.tf = list(paths), list(labels), tf

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return self.tf(Image.open(self.paths[i]).convert("RGB")), self.labels[i]


def build_model(pretrained=True):
    weights = None
    if pretrained:
        try:
            weights = models.ResNet18_Weights.DEFAULT
            m = models.resnet18(weights=weights)
        except Exception as e:  # no internet connection
            print("WARNING: could not download pretrained weights, training from scratch:", e)
            m = models.resnet18(weights=None)
    else:
        m = models.resnet18(weights=None)
    m.fc = nn.Linear(m.fc.in_features, NUM_CLASSES)
    return m


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    preds = []
    for x, _ in loader:
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            preds.append(model(x.to(device, non_blocking=True)).argmax(1).cpu())
    return torch.cat(preds).numpy()


def run(split, df, epochs, batch, workers, pretrained):
    print(f"\n=== resnet18_transfer  |  split = {split} ===")
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        print("WARNING: no GPU found. This will be very slow; enable a GPU runtime.")
    y, drivers, paths = df.label.values, df.subject.values, df.path.values
    tr, va = make_split(y, drivers, split)

    dl_tr = DataLoader(Frames(paths[tr], y[tr], train_tf), batch_size=batch, shuffle=True,
                       num_workers=workers, pin_memory=True, drop_last=True)
    dl_tr_eval = DataLoader(Frames(paths[tr], y[tr], eval_tf), batch_size=batch * 2,
                            num_workers=workers, pin_memory=True)
    dl_va = DataLoader(Frames(paths[va], y[va], eval_tf), batch_size=batch * 2,
                       num_workers=workers, pin_memory=True)

    model = build_model(pretrained).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-4, total_steps=epochs * len(dl_tr))
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)

    hist = {"loss": [], "val_acc": []}
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        running = 0.0
        for x, t in dl_tr:
            x, t = x.to(device, non_blocking=True), t.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                loss = loss_fn(model(x), t)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            running += loss.item()
            hist["loss"].append(loss.item())
        va_acc = float((predict(model, dl_va, device) == y[va]).mean())
        hist["val_acc"].append(va_acc)
        print(f"  epoch {ep + 1}/{epochs}  loss {running / len(dl_tr):.4f}  val acc {va_acc:.4f}"
              f"  ({time.time() - t0:.0f}s)")
    secs = time.time() - t0

    metrics = evaluate(y[va], predict(model, dl_va, device), y[tr], predict(model, dl_tr_eval, device))
    rec = save_result("resnet18_transfer", split, metrics,
                      {"hyperparameters": {"epochs": epochs, "batch": batch, "lr": 3e-4,
                                           "optimizer": "AdamW + OneCycle", "img_size": IMG,
                                           "pretrained": pretrained},
                       "train_seconds": round(secs, 1)})
    torch.save(model.state_dict(), MODELS_DIR / f"resnet18_transfer_{split}.pt")
    plot_confusion(metrics["confusion_matrix"], f"ResNet-18 transfer ({split} split)",
                   figure_path(f"cm_resnet18_transfer_{split}.png"))
    plot_curve(hist["val_acc"], f"ResNet-18 validation accuracy per epoch ({split})",
               "accuracy", figure_path(f"acc_resnet18_transfer_{split}.png"))
    return rec


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+", default=list(SPLITS), choices=list(SPLITS))
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--no_pretrained", action="store_true")
    ap.add_argument("--data_dir", default=None)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    df = load_index(a.data_dir)
    for split in a.splits:
        if not already_done("resnet18_transfer", split, a.force):
            run(split, df, a.epochs, a.batch, a.workers, not a.no_pretrained)
