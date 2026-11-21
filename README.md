# Signer-independent sign language recognition from hand landmarks (Slovo)

Isolated sign recognition on the **Slovo** Russian Sign Language dataset (1,000 signs, 194 signers), using only the **hand landmarks** shipped with the dataset
and a strict **signer-disjoint** evaluation. Everything runs on free **Google Colab**. The project is a step-by-step ladder (classical models, MLP, 1D-CNN, GRU,
augmentation, ensembles) with honest statistics, an audit of the dataset's official split, and an error analysis.

> **Research question.** Which input cues and model families generalize best to signers the model has never seen?
> **Status.** Hands-only study finished. Next step (not implemented): add **face and body** landmarks, because many signs combine manual and non-manual signals.

## Key results (300-sign development subset, 5-fold signer-disjoint cross-validation, top-1; chance = 0.33 %)

| Model | Accuracy |
|---|---|
| Linear SVM on 648 summary features | 46.8 ± 3.1 % |
| MLP on summary features | 53.0 ± 3.7 % |
| 1D-CNN on the raw landmark sequence | 63.6 ± 4.0 % |
| BiGRU | 60.9 ± 4.0 % |
| 1D-CNN + rotation augmentation + robust normalization (mean of 2 seeds) | 66.4 ± 3.0 % |
| **Ensemble of two CNN seeds + one GRU** | **69.7 ± 4.1 %** (95 % interval over signers 67.0 to 72.6) |

Top-2 / top-5 accuracy of the ensemble: 82.9 % / 92.1 %. All key numbers of every stage are collected in [`results/RESULTS_SUMMARY.pdf`](results/RESULTS_SUMMARY.pdf). Answering only the 50 % most confident clips gives 88.5 % accuracy.

## What we found (the interesting parts)

- **The official train/test split of Slovo leaks signers:** 92 of 194 signers appear in both train and test. We built our own signer-disjoint split.
- **Seven signers recorded 68 % of all clips.** A naive signer split makes one person most of the test set. Prolific signers are always in training, the rest are balanced over 5 folds.
- **Data quirks fixed:** the two hand slots ("hand 1/2") are detection order and swap between frames (tracking cut large jumps from 3.6 % to 0.2 % of frames); about 7 % of frames are missing from the landmark file; x and y need an aspect-ratio correction.
- **Normalization by a single projected bone collapses in about 5 % of frames** (values up to 294). A robust 3D four-bone scale helps by about 2 points. Clipping alone does not.
- **Structure matters most:** the same raw input gives 45 % with a flattened MLP and 66 % with a CNN.
- **Single-run cross-validation is noisy (about 2 points).** An early estimate of +4 from one run shrank to +2 after repeating it; average several seeds before claiming small gains.
- **Signer leakage grows with model flexibility** (seen minus unseen signers: +0.5 points for logistic regression, +3 for the MLP, roughly +4 to +8 for the CNN and GRU).
- **Errors concentrate on related words, pointing signs and one-hand signs** (61.6 % vs 74.2 % for two-hand signs), which suggests missing non-manual information.

## Repository layout

```
notebooks/   00 to 08, run in order (see the table)
docs/        PROJECT_HANDOFF.md: full context, data findings, protocol, every result, pitfalls and roadmap
results/     RESULTS_SUMMARY.pdf (all key numbers on 4 pages) and, once you run the notebooks, the CSV result files copied from your Drive
figures/     put the figures here
```

| Notebook | Purpose | Runtime (Colab) |
|---|---|---|
| `00_data_preparation` | download landmarks, audit the split, hand tracking, signer-disjoint split, subset, cached arrays | CPU, about 20 min |
| `01_classical_baselines` | 648 summary features; kNN, PCA + logistic regression, linear SVM, random forest; leakage test | CPU, 20 to 40 min (SVM is slow) |
| `02_mlp` | first PyTorch network, regularization, seeds, CV | GPU, 10 to 20 min |
| `03_cnn1d` | 1D-CNN on the raw sequence, tuning on validation | GPU, 20 to 40 min |
| `04_gru` | (Bi)GRU, seed ensemble | GPU, 30 to 50 min |
| `05_augmentation_ensembling` | seven augmentations, recipe selection, CNN and GRU, ensembles | GPU, about 90 min |
| `06_robust_normalization` | four hand-size normalizations compared | GPU, about 40 min |
| `07_final_pipeline` | robust normalization + rotation, CNN and GRU, cross-validated ensembles, saves predictions | GPU, about 65 min |
| `08_error_analysis` | per-sign, confusions, clip properties, signers, confidence (no training) | CPU, 1 to 2 min |

Runtimes are approximate. Inside `docs/PROJECT_HANDOFF.md` and inside the notebooks, they are also called Stage 0, 1, 2, 3, 4, 4b, 4c, final pipeline and Stage 7a.

## How to run

1. Open a notebook in Colab (`https://colab.research.google.com/github/<USER>/<REPO>/blob/main/notebooks/00_data_preparation.ipynb`).
2. All notebooks read and write a Google Drive folder, `/content/drive/MyDrive/slovo_project/` (variable `BASE` in the first code cell; change it if you prefer another place).
3. Run the notebooks **in order, each in a fresh session, cells from the top**. Each stage reads the files written by the earlier ones. Mixing cells from different notebooks or skipping the first cell caused most of our errors.
4. Notebook `00` downloads the landmark file (1.2 GB) from the dataset host and reads `annotations.csv` out of the remote zip. The host is in Russia and can be slow; fallbacks are printed by the notebook.
5. Never change the seed (`42`), the folds or the 300-sign subset once results exist, or later stages stop being comparable.

## Evaluation protocol (short)

No signer is shared between train, validation and test; tuning uses the validation fold only; normalization statistics come from training data only;
5-fold CV rotates the test fold; results are quoted as mean ± standard deviation over folds with paired per-fold comparisons; differences below about 3 points are treated as noise.
The full protocol is in `docs/PROJECT_HANDOFF.md`, section 6.

## Data and license

- The **Slovo dataset is not included** and is not redistributed here. Get it from https://github.com/ai-forever/slovo and read its license (a CC BY-SA 4.0 variant). Please cite: Kapitanov et al., *Slovo: Russian Sign Language Dataset*, arXiv:2305.14527.
- Processed arrays are derived from the dataset and are **not committed** (see `.gitignore`); the notebooks regenerate them.
- The code and notebooks are under the MIT license (`LICENSE`).

## Limitations

Hands only (no face or body), about 17 training clips per sign in the subset, extreme signer imbalance, one seed per fold in the CV, hyper-parameters chosen on validation fold 1, a confounded seen-versus-unseen leakage test. Details in `docs/PROJECT_HANDOFF.md`, section 14 (and 6.4).

## Roadmap / how to contribute

1. **Face and body landmarks** (the main next step): extract face, pose and hands from the videos in one pass (hands too, because the shipped hand file has no frame indices), with a pilot on about 50 clips first.
   Expected issues: the legacy MediaPipe Holistic API may be missing in recent versions (use the Tasks landmarkers or pin a version); body-centred robust normalization; hand position relative to the face and chest; identity leakage through face geometry (prefer expression scores and subtract a neutral face); hands occluding the face; per-modality missing flags; more overfitting with 3x more inputs (modality dropout).
   Evaluate as an ablation (hands, hands + body, hands + face, all) with the same CNN, folds and seeds.
2. Cheap checks on the saved predictions: clip length within a sign, a signer handedness proxy (then mirror augmentation), hubness correction, calibration.
3. Transformer with one token per body part, and a skeleton graph model.
4. Final comparison on all 1,000 signs and a controlled leakage experiment (add k clips per test signer to the training set).

## Citation

See `CITATION.cff` (replace `<USER>/<REPO>` after publishing).
