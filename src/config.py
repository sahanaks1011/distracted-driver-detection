"""Shared constants and paths for the whole project."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# On Colab we set DDD_OUT to a shared Google Drive folder so results survive the session
# and both teammates write into the same place. Locally it defaults to the repo folder.
OUT = Path(os.environ.get("DDD_OUT", ROOT))
CACHE_DIR = ROOT / "data" / "cache"
RESULTS_DIR = OUT / "results"
FIG_DIR = RESULTS_DIR / "figures"
MODELS_DIR = OUT / "models"

for _d in (CACHE_DIR, RESULTS_DIR, FIG_DIR, MODELS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

SEED = 42
IMG_SIZE = 64          # paper: 640x480 -> 64x64x3 -> 12288-dim vector
VAL_FRAC = 0.20        # paper: 80 / 20 split
SPLITS = ("random", "driver")

CLASSES = [
    "c0 Safe driving",
    "c1 Texting (right)",
    "c2 Phone call (right)",
    "c3 Texting (left)",
    "c4 Phone call (left)",
    "c5 Operating radio",
    "c6 Drinking",
    "c7 Reaching behind",
    "c8 Hair and makeup",
    "c9 Talking to passenger",
]
SHORT = ["safe", "text_R", "call_R", "text_L", "call_L",
         "radio", "drink", "reach", "makeup", "passenger"]
NUM_CLASSES = len(CLASSES)
