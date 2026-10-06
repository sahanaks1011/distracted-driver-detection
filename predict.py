"""Live demo: classify driver images with a trained model.

Owner: Rachel Sara Mathew
Run:
  python -m src.predict --export 12                     # copy 12 validation images to OUT/demo_images
  python -m src.predict --folder demo                   # predict every image in OUT/demo_images (no dataset needed)
  python -m src.predict --random 8                      # 8 random validation images (needs the dataset)
  python -m src.predict --image some_photo.jpg          # any image file
Options: --model nn|cnn  --split random|driver   (which saved model to load)
"""
import argparse

import numpy as np

import shutil
from pathlib import Path

from .config import CLASSES, FIG_DIR, IMG_SIZE, MODELS_DIR, OUT, SEED, SHORT
from .data import load_index, make_split, read_image
from .models_scratch import TwoLayerNet


class NNPredictor:
    def __init__(self, split):
        self.net, self.mean, self.std = TwoLayerNet.load(MODELS_DIR / f"two_layer_net_{split}.npz")

    def proba(self, path):
        x = read_image(path, IMG_SIZE).reshape(1, -1).astype(np.float32)
        return self.net.predict_proba((x - self.mean) / self.std)[0]


class CNNPredictor:
    def __init__(self, split):
        import torch
        from .train_cnn import build_model, eval_tf
        self.torch, self.tf = torch, eval_tf
        self.dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.m = build_model(pretrained=False)
        self.m.load_state_dict(torch.load(MODELS_DIR / f"resnet18_transfer_{split}.pt",
                                          map_location=self.dev))
        self.m.to(self.dev).eval()

    def proba(self, path):
        from PIL import Image
        x = self.tf(Image.open(path).convert("RGB")).unsqueeze(0).to(self.dev)
        with self.torch.no_grad():
            return self.torch.softmax(self.m(x), 1)[0].cpu().numpy()


def get_predictor(model, split):
    return NNPredictor(split) if model == "nn" else CNNPredictor(split)


def show(path, p, true=None):
    top = np.argsort(p)[::-1][:3]
    head = f"{path}" + (f"   [true: {CLASSES[true]}]" if true is not None else "")
    print(head)
    for k in top:
        print(f"    {CLASSES[k]:26s} {100 * p[k]:5.1f}%")


def grid(items, pred, out):
    """items: list of (path, true_label or None). Saves a picture grid with predictions."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cols = min(4, len(items))
    rows = int(np.ceil(len(items) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.3 * cols, 3 * rows), squeeze=False)
    correct = labelled = 0
    for ax, (path, t) in zip(axes.flat, items):
        p = pred.proba(path)
        k = int(np.argmax(p))
        show(path, p, t)
        ax.imshow(read_image(path, 256))
        title = f"pred {SHORT[k]} {100 * p[k]:.0f}%"
        colour = "black"
        if t is not None:
            labelled += 1
            correct += k == t
            title += f"\ntrue {SHORT[t]}"
            colour = "green" if k == t else "red"
        ax.set_title(title, fontsize=9, color=colour)
        ax.axis("off")
    for ax in axes.flat[len(items):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    if labelled:
        print(f"\n{correct}/{labelled} correct.")
    print(f"Grid saved to {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="nn", choices=["nn", "cnn"])
    ap.add_argument("--split", default="driver", choices=["random", "driver"])
    ap.add_argument("--image", nargs="*")
    ap.add_argument("--random", type=int, default=0)
    ap.add_argument("--export", type=int, default=0)
    ap.add_argument("--folder", default=None, help="'demo' means OUT/demo_images")
    ap.add_argument("--data_dir", default=None)
    ap.add_argument("--seed", type=int, default=SEED)
    a = ap.parse_args()
    demo_dir = OUT / "demo_images"

    if a.export:   # copy validation images of unseen drivers so the demo works without the dataset
        df = load_index(a.data_dir)
        _, va = make_split(df.label.values, df.subject.values, "driver")
        demo_dir.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(a.seed)
        for c in range(10):
            pool = [i for i in va if df.label[i] == c]
            for i in rng.choice(pool, max(1, a.export // 10), replace=False):
                shutil.copy(df.path[i], demo_dir / f"c{c}_{df.img[i]}")
        print(f"Copied {len(list(demo_dir.glob('*.jpg')))} images to {demo_dir}")
        raise SystemExit

    pred = get_predictor(a.model, a.split)
    for path in a.image or []:
        show(path, pred.proba(path))

    if a.folder:
        folder = demo_dir if a.folder == "demo" else Path(a.folder)
        files = sorted(folder.glob("*.jpg")) + sorted(folder.glob("*.png"))
        items = []
        for f in files:
            t = int(f.name[1]) if f.name[0] == "c" and f.name[1].isdigit() else None
            items.append((str(f), t))
        grid(items, pred, FIG_DIR / f"demo_{a.model}_{a.split}.png")

    if a.random:
        df = load_index(a.data_dir)
        _, va = make_split(df.label.values, df.subject.values, a.split)
        pick = np.random.default_rng(a.seed).choice(va, a.random, replace=False)
        grid([(df.path[i], int(df.label[i])) for i in pick], pred,
             FIG_DIR / f"demo_{a.model}_{a.split}.png")
