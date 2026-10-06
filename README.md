# Distracted Driver Detection

UE24CS352A Machine Learning Mini-Project, Section G, Team 7, Problem Statement 75

| Member | SRN | Main responsibility |
|---|---|---|
| Sahana K S (Part A) | PES2UG24CS425 | Data pipeline, from-scratch NumPy models (Linear SVM, Softmax, Two-layer Net), tuning |
| Rachel Sara Mathew (Part B) | PES2UG24CS390 | scikit-learn baselines, ResNet-18 extension, evaluation, report figures, demo |

Reference paper: D. Feng and Y. Yue, *Machine Learning Techniques for Distracted Driver Detection*, CS229 Spring 2019
([report](https://cs229.stanford.edu/proj2019spr/report/24.pdf), [poster](https://cs229.stanford.edu/proj2019spr/poster/24.pdf)).

## Problem

Classify a dashboard-camera image of a driver into one of 10 activities:
c0 safe driving, c1 texting right, c2 phone right, c3 texting left, c4 phone left,
c5 radio, c6 drinking, c7 reaching behind, c8 hair and makeup, c9 talking to passenger.

## Dataset

[State Farm Distracted Driver Detection](https://www.kaggle.com/competitions/state-farm-distracted-driver-detection)
(Kaggle competition data, accept the rules on the competition page first).
22,424 labelled 640x480 images, 10 classes, 26 drivers. `driver_imgs_list.csv` maps every image to its driver.

## What we did

1. **Reproduced the paper**: images resized to 64x64x3 and flattened to 12,288 features; five classifiers:
   Linear SVM, Softmax and a 100-unit Two-layer Net **written from scratch in NumPy** (gradient-checked),
   plus Gaussian Naive Bayes and a Decision Tree from scikit-learn.
2. **Driver-wise evaluation (our addition)**: the paper's shuffled 80/20 split puts near-identical frames of the
   same driver in both train and validation. We also evaluate on an 80/20 split where no driver is shared.
3. **CNN extension (paper's future work)**: ResNet-18 pretrained on ImageNet, fine-tuned for 4 epochs.
4. Hyperparameters are tuned on a tuning split carved from the training set; the validation set is used once.

## Results

Validation accuracy (%). Filled in from `results/summary.md` after running.

| Model | Paper val % | Random train % | Random val % | Driver train % | Driver val % |
|---|---|---|---|---|---|
| Naive Bayes | 54.99 | 55.92 | 54.49 | 56.06 | 27.56 |
| Decision Tree | 84.73 | 100.00 | 89.25 | 100.00 | 22.60 |
| Linear SVM | 71.39 | 98.36 | 97.70 | 92.09 | 36.86 |
| Softmax | 82.31 | 85.24 | 84.01 | 73.17 | 30.47 |
| Two-layer Net | 92.24 | 99.96 | 99.35 | 98.37 | 35.28 |
| ResNet-18 (extension) | - | 99.99 | 99.78 | 99.98 | 91.46 |

Figures: `results/figures/` (accuracy comparison, confusion matrices, learning curves, demo grids).
Write-up: `results/Team7_Writeup_Distracted_Driver_Detection.pdf`.

## How to run (Google Colab)

The work is split into two notebooks that write into one shared Google Drive folder.

| Notebook | Person | Runtime | Time |
|---|---|---|---|
| `notebooks/Part_A_Sahana.ipynb` | Sahana | CPU or GPU | about 40 min |
| `notebooks/Part_B_Rachel.ipynb` | Rachel | T4 GPU | about 90 min |

1. Put `distracted-driver-detection.zip` in a Google Drive folder `MyDrive/DDD_Team7`, shared by both members
   (the second member adds it to her own My Drive with Organize > Add shortcut).
2. Get `kaggle.json` (kaggle.com > Settings > API > Create New Token) and accept the competition rules.
3. Open the notebook in Colab (File > Upload notebook), set the runtime, then Runtime > Run all.

Everything lands in `MyDrive/DDD_Team7/outputs/` (results, figures, models, write-up PDF, demo images).
Rerunning skips models that are already finished, so a Colab disconnect costs nothing.

### Command line (any machine, from the repo root)

```bash
pip install -r requirements.txt
python -m src.models_scratch                 # gradient check of the NumPy models
python -m src.data [--data_dir PATH]         # read + resize + cache images, show splits
python -m src.train_scratch                  # Part A: Linear SVM, Softmax, Two-layer Net (both splits)
python -m src.train_sklearn                  # Part B: Naive Bayes, Decision Tree (both splits)
python -m src.train_cnn                      # Part B: ResNet-18 transfer learning (GPU)
python -m src.report                         # comparison table + figures
python -m src.make_writeup --repo_url URL    # two-page PDF write-up
python -m src.predict --export 20            # save demo images of unseen drivers
python -m src.predict --folder demo          # live demo on those images
python app.py                                # optional Gradio web demo (pip install gradio)
```

`PATH` is the folder containing `driver_imgs_list.csv` and `imgs/train/c0..c9`.
Set the environment variable `DDD_OUT` to choose where results and models are written.
Useful flags: `--splits random driver`, `--models ...`, `--quick`, `--force` (retrain), `--epochs N`.
Dry run without real data: `python -m src.make_fake_dataset`.

## Repository layout

```
src/config.py           constants, class names, paths
src/data.py             loading, caching, random and driver-wise splits, standardisation   (Part A)
src/models_scratch.py   LinearSVM, Softmax, TwoLayerNet in NumPy + gradient check          (Part A)
src/train_scratch.py    grid search + training of the NumPy models                          (Part A)
src/train_sklearn.py    Naive Bayes, Decision Tree                                          (Part B)
src/train_cnn.py        ResNet-18 transfer learning in PyTorch                              (Part B)
src/evaluate.py         metrics, confusion matrices, result files                           (Part B)
src/report.py           summary table and comparison plots                                  (Part B)
src/predict.py, app.py  live demo                                                           (Part B)
src/make_writeup.py     PDF write-up generator                                              (Part A + B)
notebooks/Part_A_Sahana.ipynb, Part_B_Rachel.ipynb   Colab notebooks
results/                metrics JSON, summary tables, figures, write-up PDF
```

Seed 42 everywhere; all models share identical train / validation indices.
