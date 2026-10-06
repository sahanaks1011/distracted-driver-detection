"""Create a tiny synthetic dataset with the same folder layout as State Farm.

Only for checking that the code runs end to end on a laptop before using the real data.
Run:  python -m src.make_fake_dataset      -> data/fake/ (then pass --data_dir data/fake)
"""
import cv2
import numpy as np
import pandas as pd

from .config import ROOT


def main(per_class_per_driver=6, drivers=26):
    out = ROOT / "data" / "fake"
    rng = np.random.default_rng(0)
    rows = []
    for d in range(drivers):
        bg = rng.integers(40, 200, 3)
        for c in range(10):
            (out / "imgs" / "train" / f"c{c}").mkdir(parents=True, exist_ok=True)
            for k in range(per_class_per_driver):
                img = np.full((480, 640, 3), bg, np.uint8)
                img = np.clip(img + rng.normal(0, 25, img.shape), 0, 255).astype(np.uint8)
                cx = 80 + 50 * c + int(rng.normal(0, 25))
                cy = 120 + 25 * (c % 4) + int(rng.normal(0, 25))
                cv2.circle(img, (cx, cy), 45, (255, 255 - 20 * c, 20 * c), -1)
                name = f"img_{d}_{c}_{k}.jpg"
                cv2.imwrite(str(out / "imgs" / "train" / f"c{c}" / name), img)
                rows.append((f"p{d:03d}", f"c{c}", name))
    pd.DataFrame(rows, columns=["subject", "classname", "img"]).to_csv(out / "driver_imgs_list.csv", index=False)
    print(f"Wrote {len(rows)} fake images to {out}")


if __name__ == "__main__":
    main()
