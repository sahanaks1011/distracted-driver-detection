"""Optional web demo for the review (upload a driver photo, get the top-3 classes).

Run:  pip install gradio
      python app.py              (prints a public *.gradio.live link when run in Colab / Kaggle)
"""
import tempfile

import gradio as gr

from src.config import CLASSES, MODELS_DIR
from src.predict import CNNPredictor, NNPredictor

predictors = {}
for split in ("driver", "random"):
    if (MODELS_DIR / f"two_layer_net_{split}.npz").exists():
        predictors[f"Two-layer Net (from scratch, {split} split)"] = NNPredictor(split)
    if (MODELS_DIR / f"resnet18_transfer_{split}.pt").exists():
        predictors[f"ResNet-18 transfer ({split} split)"] = CNNPredictor(split)
if not predictors:
    raise SystemExit("No trained models in models/. Train first.")


def classify(image, model_name):
    with tempfile.NamedTemporaryFile(suffix=".jpg") as f:
        image.save(f.name)
        p = predictors[model_name].proba(f.name)
    return {CLASSES[i]: float(p[i]) for i in range(len(CLASSES))}


demo = gr.Interface(
    fn=classify,
    inputs=[gr.Image(type="pil", label="Dashboard camera frame"),
            gr.Dropdown(list(predictors), value=list(predictors)[0], label="Model")],
    outputs=gr.Label(num_top_classes=3, label="Prediction"),
    title="Distracted Driver Detection",
    description="UE24CS352A mini-project, Team 7 (Sahana K S, Rachel Sara Mathew)",
)

if __name__ == "__main__":
    demo.launch(share=True)
