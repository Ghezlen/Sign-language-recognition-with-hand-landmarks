> **Note on file names.** In this repository the notebooks are numbered `00` to `08`. Inside this document (and inside the notebooks) they are called by their
> original working names: `00` = Stage 0 (`stage0_slovo_data`), `01` = Stage 1, `02` = Stage 2, `03` = Stage 3, `04` = Stage 4, `05` = Stage 4b (augmentation and
> ensembling), `06` = Stage 4c (robust normalization), `07` = final pipeline, `08` = Stage 7a (error analysis). The written report mentioned in section 4.2 is not part of the
> repository (export it separately if you want to include it under `docs/`).

# Signer-independent sign language recognition from hand landmarks — project handoff

This document gives a new team member everything needed to understand the project, reproduce the work done so far, and continue it.
It covers: the goal, the data, every decision and why it was made, the pipeline and files, all results so far, the evaluation rules we follow,
problems we hit, and the concrete next steps.

**Status at the time of writing:** Stages 0 to 4, the augmentation + ensembling step (Stage 4b) and the robust-normalization experiment (Stage 4c) are finished and were run on Google Colab. Stage 5 (Transformer) has not been written yet (see section 12).

---

## Contents

1. [The project in one page](#1-the-project-in-one-page)
2. [Background and how the idea was chosen](#2-background-and-how-the-idea-was-chosen)
3. [The dataset (Slovo)](#3-the-dataset-slovo)
4. [Environment, files and how to run everything](#4-environment-files-and-how-to-run-everything)
5. [Stage 0: data preparation and the evaluation split](#5-stage-0-data-preparation-and-the-evaluation-split)
6. [Evaluation protocol (rules every experiment must follow)](#6-evaluation-protocol-rules-every-experiment-must-follow)
7. [Stage 1: classical baselines](#7-stage-1-classical-baselines)
8. [Stage 2: MLP](#8-stage-2-mlp)
9. [Stage 3: 1D-CNN](#9-stage-3-1d-cnn)
10. [Stage 4: GRU](#10-stage-4-gru)
11. [All results side by side, and what we learned](#11-all-results-side-by-side-and-what-we-learned)
12. [What remains: the roadmap](#12-what-remains-the-roadmap)
13. [Mistakes and pitfalls (read this before touching anything)](#13-mistakes-and-pitfalls-read-this-before-touching-anything)
14. [Limitations and open questions](#14-limitations-and-open-questions)
15. [References](#15-references)
16. [Glossary](#16-glossary)

---

## 1. The project in one page

**Task.** Isolated sign language recognition: given a short clip of one sign, predict which sign it is out of a fixed vocabulary.

**Input.** We do not use video. We use hand landmarks (21 points per hand, 3 coordinates each) that were already extracted from every frame
of the videos with MediaPipe by the dataset authors.

**Dataset.** Slovo, a Russian Sign Language dataset: 20,400 clips, 1,000 signs, 194 signers.

**Research question.** Which input cues and model families generalize best to signers the model has never seen, and how does the diversity of
signers in the training set affect that generalization?

**Positioning.** This is an empirical study under a strict, reproducible, signer-disjoint evaluation. It is not a new state-of-the-art model, and
"signer-invariant learning" is not claimed as a new idea (see section 15, related work). The value we add is:
- a fully documented signer-disjoint protocol on a large public dataset with skeleton input (the dataset's own published baselines are all video models);
- an audit of the official split (it is **not** signer-disjoint, see section 3);
- a measurement of how much signer leakage inflates accuracy, as a function of model flexibility;
- a controlled comparison of classical, MLP, CNN and recurrent models (Transformer planned) with the same folds, seeds and statistics;
- everything on free Colab compute.

**Secondary goal.** The project owner is a Master's student (Data Science, Big Data) who also wants to learn deep learning by building each model
family in turn. That is why the work is organised as a ladder of stages and why each stage includes sanity checks and explanations.

**Where we are in one sentence.** A 1D-CNN on the raw landmark sequence is our best single model (63.6 ± 4.0 % top-1 in 5-fold signer-disjoint cross-validation on a 300-sign development subset, 66.0 ± 4.7 % with rotation augmentation, chance 0.33 %), an ensemble of an augmented CNN and an augmented GRU reaches 67.7 ± 3.7 %, fixing the hand-size normalization added about 2 points (final CNN 66.4 ± 3.0 % with two seeds averaged per fold), the best cross-validated result is an ensemble of two CNN seeds and one GRU with 69.7 ± 4.1 %, and the more flexible the model, the more it seems to benefit from having seen the same signer in training.

---

## 2. Background and how the idea was chosen

1. The owner first found an IEEE paper (document 10456765, could not be opened) and the Kaggle "Pakistan Sign Language Dataset (OpenPose)"
   (static images of 37 alphabet signs and 12 words, no time dimension) and was asked for a deep learning project idea. A skeleton-based idea on
   that dataset was proposed and **rejected by the owner**.
2. Four alternative ideas were researched with surveys and dataset availability checks:
   (1) signer-independent isolated sign recognition on Slovo with skeleton and RGB fusion; (2) lightweight real-time recognition for mobile/edge;
   (3) transfer learning to a low-resource sign language (Arabic), where dataset access was uncertain (KArSL, KSU-ArSL download problems);
   (4) continuous recognition/translation (heavy compute).
3. **Idea 1 was chosen.** Because compute is limited (free Colab), it was reduced to a skeleton-first study using the landmarks that ship with Slovo;
   RGB fusion was dropped.
4. The scope was later scaled down further for a learning-oriented pipeline (stages 0 to 7). The development subset started as 100 signs and
   became 300 signs after Stage 0 showed that 100 signs give too small validation and test folds (section 5).
5. The owner's three inspiring articles were: IEEE 10526274 (Tao et al., IEEE Access 2024, a review), the CMES review by Zhang and Jiang (2024,
   science-direct id S1526149224001279) and IEEE 10670380 (**never identified**: the IEEE pages did not open and searches did not surface it;
   the report still has a placeholder reference [5] for it; ask the owner for its title).

---

## 3. The dataset (Slovo)

Source: Kapitanov et al., "Slovo: Russian Sign Language Dataset", ICCVS 2023, arXiv:2305.14527, code and data: https://github.com/ai-forever/slovo.
The license is a CC BY-SA 4.0 variant: read the exact terms before publishing anything.

### 3.1 What the dataset contains

| Item | Value |
|---|---|
| Videos | 20,400 (RGB, average sign clip about 50 frames) |
| Signs | 1,000, with exactly 20 clips each |
| Extra class | `no_event` (400 clips, no sign). **It is not one of the 1,000 signs and we set it aside** |
| Signers | 194 (column `user_id`) |
| Landmarks | `slovo_mediapipe.json` (about 1.2 GB), MediaPipe hand landmarks for every frame of the trimmed videos |
| Annotations | `annotations.csv` with, among others: `attachment_id`, `user_id`, `width`, `height`, `length`, `text` (the sign), `train` (official split flag) |

Published baselines in the dataset repository are video models only (MViTv2, Swin, ResNet-i3d, reported scores roughly 33 to 64). There is no skeleton baseline.

### 3.2 The landmark file format (found by inspection, not documented)

- Top level: a dictionary `clip_id -> list of frames`.
- Each frame: a dictionary with one or two entries `"hand 1"` and `"hand 2"`.
- Each hand: a list of 21 points, each a dictionary `{x, y, z}`. x and y are **fractions of the image width and height**; z is a relative depth.
- Frames in which no hand was detected are **left out of the file** (there is no frame index). Evidence: landmark frames never exceed the annotated
  video length, 74.7 % of clips have fewer landmark frames than video frames (median gap 1, 95th percentile 13, max 168), on average 6.7 % of
  video frames are absent, and the 10 clips with no detection at all are empty lists.
- `"hand 1"` and `"hand 2"` mean **detection order, not left and right**. Without correction a physical hand jumps between slots.

### 3.3 Findings about the data that shaped the whole project

| Finding | Numbers |
|---|---|
| The official train/test split is **not** signer-disjoint | 112 signers in train, 174 in test, **92 in both**. Of 4,998 official test clips (with landmarks), 1,609 come from signers never seen in the official train set |
| A few signers recorded most of the data | 7 signers recorded 13,646 of 19,990 clips (**68 %**): 3,671, 3,448, 2,635, 1,868, 1,008, 616 and 400 clips (the 8th signer has 218) |
| Empty clips | 10 clips have no landmarks and are skipped |
| Size after cleaning | 19,990 sign clips, 935,398 frames, mean 46.8 frames per clip (median 45, 95th percentile 74, 99th percentile 98) |
| Hands per frame | 36.7 % of frames have one hand, 63.3 % two |
| Low-coverage clips | 209 clips (about 1 %) miss more than half of their video frames; they are flagged (`low_coverage`) but kept |
| Aspect ratio | Videos are not square, so x and y are not in the same unit unless corrected with width/height |
| Raw detection errors | Individual frames contain glitches (hand skeletons stretched across the image) |
| Hand-size scale | The scale used to normalize each frame (wrist to middle-finger base, image plane) has percentiles 1/5/50/95 of 0.014/0.028/0.060/0.125 image heights; frames with the smallest scales produce all the extreme normalized values (see section 10c) |

---

## 4. Environment, files and how to run everything

### 4.1 Environment

- Everything runs in **Google Colab** (free tier). Stages 0 and 1 run on CPU. Stages 2 to 4 need a GPU runtime (T4).
- Python libraries used: numpy, pandas, matplotlib, scikit-learn, PyTorch (preinstalled on Colab), ijson, remotezip, tqdm.
- All persistent data lives in Google Drive under `/content/drive/MyDrive/slovo_project/` (folder names below).
- **Each stage notebook is standalone**: it reads files from Drive written by earlier stages. Open each in a **fresh Colab session** and run its cells in
  order from the first code cell. Do not mix cells between notebook versions (this caused several failures, see section 13).

### 4.2 Notebooks (deliverables of the project, send them together with this file)

| File | What it does |
|---|---|
| `stage0_slovo_data.ipynb` | Downloads data, audits the official split, converts landmarks, tracks hands, builds the signer-disjoint split and the subset, writes all cached files |
| `stage1_classical_baselines.ipynb` | 648 summary features, kNN, PCA + logistic regression, linear SVM, random forest, three small experiments |
| `stage2_mlp_pytorch.ipynb` | MLP on features and on the flattened raw sequence, regularization study, seeds, 5-fold CV, leakage test |
| `stage3_cnn1d.ipynb` | 1D-CNN on the raw sequence, tuning on validation, seeds, CV, leakage test |
| `stage4_gru.ipynb` | (Bi)GRU, tuning, seed ensemble, CV, leakage test |
| `stage4b_augmentation_ensembling.ipynb` | Data augmentation (7 types) tuned on the CNN, then applied to the CNN and the GRU; seed and CNN+GRU ensembles; 5-fold CV and leakage re-test. Results in section 10b |
| `stage4c_robust_normalization.ipynb` | Tests four hand-size normalization variants (current, clipping at +/-5, a robust 3D palm-bone scale, both) with the CNN + rotation recipe: 3 seeds and 5-fold CV with paired comparisons. Results in section 10c |
| `final_pipeline_robust_ensembles.ipynb` | Final pipeline: robust3d normalization + rotation augmentation; CNN (3 seeds, CV with 2 seeds per fold, leakage), GRU (3 seeds, CV, leakage), cross-validated ensembles (two CNN seeds vs CNN + GRU), consolidated table; saves cross-validated logits (`results/final_cv_logits.npz`) for error analysis. Results in section 10d |
| `stage7a_error_analysis.ipynb` | Error analysis on the saved cross-validated predictions (CPU, no training): bootstrap interval over signers, per-sign accuracy, most confused pairs and sink signs, accuracy by clip length / missing frames / low coverage / one-hand vs two-hand, per-signer accuracy, model agreement, confidence calibration and coverage-accuracy curve. Results in section 10e |

The written report is a Claude Doc titled "Signer-Independent Sign Language Recognition from Skeleton Keypoints — Report, Part 1 (Stage 0)".
It currently has sections 1 to 10 (introduction, state of the art, motivation, scope, pipeline, Stage 0 protocol and findings, and one section per
stage 1 to 4) plus the reference list. It is private to the owner unless shared.

### 4.3 Drive layout and files produced

```
slovo_project/
  raw/          slovo_mediapipe.json (1.2 GB), annotations.csv
  processed/    splits.csv, landmarks_raw.npz, frames_tracked_v2.npy,
                features_stage1_subset_T64_S3.npz, sequences_subset_T64.npz,
                subset_classes.csv, signers.csv, no_event_ids.csv, empty_clips.csv, stage0_summary.json
  figures/      stage0_*.png, stage1_results.png, stage2_*.png, stage3_*.png, stage4_*.png
  results/      stage1_results.csv, stage2_results.csv, stage3_results.csv, stage3_sweeps.csv,
                stage4_results.csv, stage4_sweeps.csv
```

(`frames_tracked.npy` and `slovo100_classes.csv` are stale leftovers from earlier versions and can be deleted.)

| File | Content |
|---|---|
| `landmarks_raw.npz` | `frames` (935,398 x 2 x 21 x 3, float16, NaN where a hand is missing), `lengths` (frames per clip), `ids` (clip ids) |
| `frames_tracked_v2.npy` | Same array after hand tracking (stable slots). Row order identical to `landmarks_raw.npz` |
| `splits.csv` | One row per clip with the columns below |
| `features_stage1_subset_T64_S3.npz` | 648 summary features per subset clip (`X`, `ids`), made by Stage 1 |
| `sequences_subset_T64.npz` | Raw sequences per subset clip, 64 x 132 (`S` float16, `ids`), made by Stage 2 and reused by Stages 3 and 4 |

`splits.csv` columns: `attachment_id`, `user_id`, `text` (sign), `label_id_full` (0 to 999), `label_id_subset` (0 to N-1 for the development subset, empty otherwise),
`in_subset`, `split` (`train`/`val`/`test`), `fold` (-1 for prolific signers, else 0 to 4), `prolific_signer`, `official_train`,
`official_test_unseen_signer`, `aspect` (width/height), `n_frames`, `missing_rate`, `low_coverage`.

### 4.4 Source of the data in Stage 0

- Landmarks: `https://rndml-team-cv.obs.ru-moscow-1.hc.sbercloud.ru/datasets/slovo/slovo_mediapipe.json` (resumable `wget`).
- `annotations.csv` is read out of the remote 16 GB `slovo.zip` with the `remotezip` library, without downloading the zip. Fallbacks (upload by hand, or the Kaggle
  dataset `kapitanov/slovo`) are printed by the notebook if this fails. The host is in Russia and may be slow or blocked from some regions.

### 4.5 Reproducing from scratch (recommended order)

1. Run Stage 0 (CPU, the JSON conversion takes about 2 minutes, hand tracking a few minutes). Or get the Drive folder `processed/` from the owner and skip it.
2. Run Stage 1 (CPU). It writes the features cache and `stage1_results.csv`. The linear SVM alone takes about 10 minutes per fit; set `RUN_SLOW_MODELS = False`
   to skip it, and keep `CV_INCLUDE_SVM = False` (the 5-fold SVM run took over an hour).
3. Run Stage 2, then 3, then 4 on a GPU runtime. Stages 3 and 4 read the CSV files of the earlier stages from `results/` to print paired per-fold comparisons.

Measured run times on a Colab T4: MLP 3 to 8 s per run; CNN 22 to 42 s per run; GRU 120 to 200 s per cross-validation run.

---

## 5. Stage 0: data preparation and the evaluation split

### 5.1 Steps in the notebook

1. Download landmarks and `annotations.csv`.
2. **Audit the official split** (found it not signer-disjoint), set `no_event` aside.
3. Stream-parse the 1.2 GB JSON with `ijson` into one array. Skip the 10 empty clips.
4. Estimate missing frames as (video length - landmark frames).
5. **Hand tracking** (cell "11b"): re-assign the two hand slots so each physical hand keeps one slot, by matching wrists between consecutive frames
   (first frame: leftmost hand to slot 0). Large wrist jumps (more than 20 % of the image between consecutive frames) fell from **3.59 % to 0.18 %** of frame pairs.
   The raw array stays unchanged (`frames_raw`); the tracked array is a separate file.
6. Build the signer-disjoint split and the development subset (below).
7. Define and unit-test the preprocessing used everywhere (below). Save all files and statistics.

### 5.2 Preprocessing used by every model

- **Aspect correction:** multiply x by width/height so x and y share a unit.
- **Normalization per frame and hand:** subtract the wrist, divide by the wrist-to-middle-finger-base distance (xy). The result describes hand *shape* and is invariant
  to the hand's position in the image and its apparent size. NaN (missing hand) stays NaN.
- **Fixed length:** every clip is resampled to **T = 64 frames** by nearest-index sampling (linspace over the clip); short clips repeat frames, long ones skip frames.
  This treats the stored frames as evenly spaced, although about 6.7 % of frames are missing (a known small distortion).
- **Missing hands:** filled with 0 and flagged by a visibility channel.
- **Standardization:** per-feature or per-channel mean and standard deviation computed **on the training rows only** (never on validation or test).
  Channels that never vary in training are left alone (standard deviation set to 1).

**Sequence tensor (Stages 2 to 4): 64 frames x 132 channels.** Channels: 0 to 125 joints (hand 0: 63 values = 21 joints x (x, y, z); then hand 1: 63),
126 to 129 wrist position xy of each hand (aspect-corrected, absolute image position), 130 and 131 visibility flags of hand 0 and hand 1.

### 5.3 The split (the most important design decision)

Why our own split: the official one leaks signers (section 3.3). Why not a simple random signer split: seven signers hold 68 % of the clips, so a naive split put one
signer into most of the validation or test set. The first signer-disjoint attempt gave validation 414 clips and test 443 clips, each dominated by a single signer.

Final design (`make_splits` in Stage 0, cell 12):

1. **Prolific signers are always in training.** The rule: any signer who would supply more than 10 % of a validation/test fold is removed from the fold pool, recomputed
   iteratively. A safeguard stops removal before the pool keeps less than 30 % of the clips. Result: **7 prolific signers** (13,646 clips) are always trained on.
2. All other signers go **entirely to one of 5 folds** with (almost) equal clip counts. Assignment: bin packing (largest signer first into the currently smallest fold, with
   seeded noise on the sizes), 60 tries, keep the one with the smallest signer dominance, then the most balanced folds, then the best sign coverage.
   Only clip counts are used, never model results. Computed on **all 1,000 signs**, so the development subset and any later full-vocabulary run share the same signer assignment.
3. **Single split:** fold 0 = test, fold 1 = validation, folds 2 to 4 + prolific signers = train.
4. **Cross-validation:** for k = 0..4 the test fold is k, the validation fold is (k+1) mod 5, the rest (including prolific signers) is train.
5. **Development subset:** 300 of the 1,000 signs, uniformly at random, seed 42 (`N_SUBSET_CLASSES = 300`; the subset is used for all model development so far).
6. The official labels stay in `splits.csv` (`official_train`, `official_test_unseen_signer`) so results can also be reported on the official split. This is how the leakage
   experiments work (section 6.4).

Resulting sizes:

| Split | All 1,000 signs: clips / signers | Development subset (300 signs): clips / signers | Largest signer share |
|---|---|---|---|
| Train | 17,452 / 121 | 5,222 / 102 | 21 % (a prolific signer) |
| Validation | 1,269 / 36 | 368 / 28 | 12 % |
| Test | 1,269 / 37 | 409 / 30 | 12 to 13 % |

In the subset, a single fold leaves 68 of 300 signs without a validation clip and 54 without a test clip, which is why results are always averaged over the 5 folds.
Fold test sizes in the subset: 409, 368, 395, 402 and 358 clips.

---

## 6. Evaluation protocol (rules every experiment must follow)

1. **No signer is shared between train, validation and test.** Never re-split randomly by clip.
2. **Same folds everywhere.** Single split = fold 0 test / fold 1 validation. Cross-validation rotates as in section 5.3. Never change `SEED`, the fold assignment or the subset once results exist.
3. **Tune only on validation.** Hyperparameters, early stopping and choices of architecture use the validation fold. The test fold is read only to report. (Known small exception: section 14.)
4. **Training statistics only.** Normalization statistics are computed on training rows.
5. **Report cross-validated numbers.** Single-split test accuracy is systematically higher than the 5-fold mean (an easier fold 0): SVM 51.3 vs 46.8, MLP 57.3 vs 53.0, CNN 66.6 vs 63.6.
   Quote the 5-fold mean ± standard deviation over folds, plus the pooled accuracy over the 1,932 test clips.
6. **Noise level.** A single validation or test fold has 368 to 409 clips, so one accuracy has a standard error of about 2.5 points; the pooled CV figure about 1.1 points; fold-to-fold
   standard deviation is 3 to 4 points; seed-to-seed standard deviation 0.5 to 1.8 points. Treat differences smaller than 3 to 4 points as noise unless they are consistent in every fold.
7. **Paired comparisons per fold.** Compare models fold by fold on the same test clips (wins out of 5, mean difference).
8. **Seeds.** Sweeps use 2 seeds per option; final numbers use 3 seeds (0, 1, 2); CV uses one seed per fold.
9. **Metrics.** top-1, top-5, macro-F1. Chance is 1/300 = 0.33 % on the subset. top-5 is not meaningful for kNN with k = 1.
10. **Training recipe for neural models.** AdamW (lr 1e-3, weight decay 0.01), batch 128, cross-entropy with label smoothing 0.1, up to 150 epochs, early stopping on validation top-1 with
    patience 20 and best weights restored. The GRU also uses gradient clipping at 1.0.
11. **Sanity check before any new network.** Train it to memorize 64 training clips; if it cannot reach near 100 %, the code is broken.

### 6.4 Signer-leakage experiment (used in every stage)

- Train on the **official training clips** of the subset (4,500 clips). Evaluate the official test clips split into **seen** signers (970 clips, signer appears in official train) and
  **unseen** signers (529 clips). The gap seen - unseen is the measured leakage. Neural models use a fixed number of epochs (median best epoch of the seed runs), no early stopping, 3 seeds.
- The standard error of a gap with these sample sizes is about 2.7 points (seed spread is much smaller than test sampling noise; clips of one signer are correlated, so the true uncertainty is larger).
- **Known confound:** seen test clips come mostly from signers with many training clips, unseen ones from signers with few, so a gap can partly reflect signer difficulty, not only memorization of signer style.

---

## 7. Stage 1: classical baselines

**Features (648 numbers per clip).** From the aspect-corrected normalized hands resampled to 64 frames:
- shape (504): per hand, the mean of the 63 joint values in each of 3 time segments (189) plus their standard deviation over the clip (63), times 2 hands;
- motion (126): per hand, mean absolute frame-to-frame change of the 63 values;
- position (18): per hand, wrist position mean, std, min, max (8 values) and, for both hands, how often the hand is visible (2 values).

**Models.** Most-frequent class; kNN (cosine, k tuned on validation: k = 1); PCA (128 components) + logistic regression (C tuned: 3.0); linear SVM (C = 0.05); random forest (150 trees).

| Model | Val top-1 | Test top-1 | Test top-5 | Test macro-F1 | 5-fold CV top-1 |
|---|---|---|---|---|---|
| Most frequent class | 0.0 | 0.0 | 0.7 | 0.0 | n/a |
| kNN (k = 1) | 35.1 | 31.3 | not meaningful | 23.9 | 31.2 ± 3.9 |
| PCA + logistic regression (C = 3) | 44.0 | 43.3 | 68.9 | 36.8 | 43.0 ± 4.2 (pooled 43.1) |
| Linear SVM (C = 0.05) | 45.9 | 51.3 | 71.9 | 42.3 | **46.8 ± 3.1** (pooled 46.9) |
| Random forest | 43.5 | 45.5 | 74.6 | 36.6 | not run |

Per-fold CV accuracies: SVM 51.8, 47.0, 42.3, 47.3, 45.8; logistic regression 46.7, 44.6, 35.9, 47.3, 40.5; kNN 32.0, 35.9, 26.1, 34.6, 27.4.
The SVM beat the logistic regression in 4 of 5 folds (tie in the fifth). Fit times: kNN under 1 s, logistic regression 55 s, SVM 612 s, forest 125 s.

**Experiments (PCA + logistic regression, val / test top-1).**

| Experiment | Result |
|---|---|
| Feature groups | shape only 41.3 / 43.3; shape + motion 40.8 / 43.3; shape + position 44.0 / 42.8; position only 3.3 / 5.6; all 44.0 / 42.5. **Hand shape carries nearly all the signal** |
| Down-weighting prolific signers (weight proportional to 1/sqrt of signer clips) | 44.0 / 43.3 became 41.3 / 41.3: no benefit |
| Leakage (official split) | logistic regression seen 38.4 vs unseen 37.8 (gap +0.5); kNN 29.8 vs 31.0 (gap -1.2): no measurable leakage |

---

## 8. Stage 2: MLP

**Model.** Linear(in, 512) -> ReLU -> Dropout -> Linear(512, 256) -> ReLU -> Dropout -> Linear(256, 300). Two inputs: the 648 standardized features, or the raw sequence flattened (64 x 132 = 8,448 inputs, about 4.5 million weights).

| Experiment (single split) | Val top-1 | Test top-1 | Test top-5 |
|---|---|---|---|
| Features, dropout 0.3 + weight decay 0.01 | 51.4 | 54.0 | 80.7 |
| Features, no regularization | 48.4 | 45.2 | 73.8 |
| Features, dropout 0.5 + weight decay 0.05 | 53.8 | 55.5 | 83.1 |
| Features, dropout 0.3 + label smoothing 0.1 (**chosen on validation**) | 55.7 | 56.0 | 81.2 |
| Raw sequence (flattened), dropout 0.5 + weight decay 0.05 | 44.8 | 44.5 | 74.1 |
| Features, chosen setting, 3 seeds | 56.4 ± 0.7 | 57.3 ± 1.8 | 82.7 |
| Raw sequence, 3 seeds | 45.4 ± 0.8 | 45.2 ± 0.5 | 75.1 |

5-fold CV of the features MLP: 56.0, 55.4, 49.9, 56.5, 47.2 -> **53.0 ± 3.7** (pooled 53.1). It beat the Stage 1 SVM in all 5 folds (mean +6.2).
Leakage: seen 49.6 ± 0.6 vs unseen 46.7 ± 0.6 (gap **+2.9**, 72 fixed epochs).

Observations: all configurations overfit (train 97 to 99 % vs validation 45 to 56 %); regularization added 5 to 7 points but did not close the gap; validation loss is lowest around epoch 10 and then rises
while validation accuracy keeps improving until about epoch 50 (the network becomes overconfident; label smoothing, which penalizes overconfidence, gave the best validation accuracy).
Hand-made summaries beat raw flattened frames by about 11 points for a plain MLP (no notion of time or hand structure).

---

## 9. Stage 3: 1D-CNN

**Input.** Sequence tensor as (channels = 132, time = 64).

**Model.** Three blocks of [Conv1d (padding = kernel // 2) -> BatchNorm -> ReLU -> MaxPool(2) -> Dropout(0.1)], then global average and global max pooling over time (concatenated), Dropout(0.5), Linear -> 300.
Receptive field after three blocks: 22 frames (kernel 3), 36 (kernel 5), 64 = the whole clip (kernel 9).

**Tuning on validation only** (greedy, one choice at a time, 2 seeds per option, val top-1 mean):

| Choice | Options -> validation top-1 |
|---|---|
| Kernel size | 3: 63.9; 5: 64.3; **9: 65.9** |
| Width | (64, 64, 128): 62.0 (264k parameters); **(128, 128, 256): 65.9 (750k)**; (256, 256, 512): 65.2 (2.4M) |
| Input channels | **all 132: 65.9**; without wrist position: 65.1; joints only: 64.8 |

Final configuration: kernel 9, widths (128, 128, 256), all channels (749,868 parameters).

| Result | Value |
|---|---|
| Baseline CNN (kernel 5, one seed): val / test / test top-5 | 63.0 / 67.2 / 91.2 |
| Tuned CNN, 3 seeds: val / test / test top-5 | 66.0 ± 1.2 / 66.6 ± 1.3 / 92.3 |
| 5-fold CV | 67.2, 63.6, 56.2, 67.2, 64.0 -> **63.6 ± 4.0** (pooled 63.7) |
| Paired vs Stage 2 MLP on features | better in 5 of 5 folds, mean +10.6 (range +6.3 to +16.8) |
| Paired vs Stage 1 SVM | better in 5 of 5 folds, mean +16.8 (range +13.9 to +19.9) |
| Leakage (71 fixed epochs) | seen 61.6 ± 0.5, unseen 57.3 ± 0.4, gap **+4.4** |

Observations: same raw input, MLP 45.2 % vs CNN 66.6 % (test) with far fewer parameters; learned filters beat hand-made summaries by about 10 points; wider windows slightly better; bigger networks not better;
position and visibility channels add nothing measurable; overfitting remains (train 99 % vs validation 60 to 63 %) but the validation loss stays flat instead of rising.

---

## 10. Stage 4: GRU

**Input.** Sequence tensor as (time = 64, channels = 132).

**Model.** Optional convolutional front-end (Conv1d kernel 5, 128 filters, BatchNorm, ReLU), a 2-layer GRU (dropout 0.3 between layers, optionally bidirectional), pooling over time (average and max, or last hidden state), Dropout(0.5), Linear -> 300.
Training adds gradient clipping (norm 1.0).

**Tuning on validation only** (greedy in this order, 2 seeds per option):

| Choice | Options -> validation top-1 |
|---|---|
| Direction | **bidirectional: 54.8**; one direction: 53.3 (277k parameters) |
| Summary over time | **average + max: 54.8**; last state: 51.4 |
| Hidden size | 64: 51.6 (228k); 128: 54.8 (652k); **256: 56.7 (2.09M)** |
| Convolutional front-end | none: 56.7; **conv: 60.6** (+3.9, the largest effect) |

Final configuration: BiGRU, hidden 256, conv front-end, average + max pooling (2,167,980 parameters).

| Result | Value |
|---|---|
| Baseline BiGRU (hidden 128, one seed): val / test / test top-5 | 54.6 / 55.0 / 87.5 |
| Tuned GRU, 3 seeds: val / test / test top-5 | 60.4 ± 0.9 / 63.7 ± 0.9 / 88.1 |
| **Ensemble of the 3 seeds** (average of probabilities): val / test / test top-5 | 63.3 / 66.7 / 91.4 (gain +2.9 / +3.1) |
| 5-fold CV | 62.8, 58.4, 56.5, 67.7, 59.2 -> **60.9 ± 4.0** (pooled 61.0) |
| Paired vs Stage 1 SVM / Stage 2 MLP | better in 5 of 5 folds each (mean +14.1 / +7.9) |
| Paired vs Stage 3 CNN | better in 2 of 5 folds, mean **-2.7** (per fold -4.4, -5.2, +0.3, +0.5, -4.7; standard error about 1.3: CNN probably slightly better, not conclusive) |
| Leakage (42 fixed epochs) | seen 57.7 ± 0.8, unseen 50.1 ± 1.6, gap **+7.6** |

Observations: recurrence did not beat the CNN and costs about 3x the parameters and 4 to 5x the time; the conv front-end helped most (the best GRU is a CNN-GRU hybrid); seed averaging gives about +3 points.

---

## 10b. Stage 4b: augmentation and ensembling

**Notebook:** `stage4b_augmentation_ensembling.ipynb`. Run time on a Colab T4: part A (tuning on the CNN) 37 minutes, CNN finals about 15 minutes, GRU part 33 minutes.

**Augmentations** (numpy, applied on raw normalized hand shapes before standardization, each to a clip with probability 0.5 and a random magnitude up to the base value; missing hands stay zero; unit-tested):
rotation (15 degrees), scale (15 %), timewarp (20 % speed, 4 frames shift), framehold (10 % of frames repeat the previous one), noise (0.03), jointdrop (5 % of joints set to 0), wristjitter (0.05). Mirroring was deliberately not tested (handedness may carry meaning).
Training for this step: AdamW, 200 epochs maximum, patience 25, label smoothing 0.1; the CNN without augmentation was re-run with identical settings.

**Recipe selection (validation only, CNN, 2 seeds):** baseline 65.8 %; each augmentation alone: rotation +0.8, timewarp +0.5, scale -0.4, jointdrop -0.4, framehold -0.8, noise -1.0, wristjitter -1.0.
Greedy combination (keep only if validation improves by more than 0.5 point): nothing improved on rotation alone. Strength x0.5 / x1 / x2: 65.8 / 66.6 / 65.4. **Final recipe: rotation, strength 1.**

| Result | CNN | GRU |
|---|---|---|
| Without augmentation, 3-seed test | 66.1 ± 2.0 (Stage 3: 66.6 ± 1.3) | 63.7 ± 0.9 (Stage 4) |
| With rotation, 3-seed test | 67.3 ± 1.0 | 65.7 ± 1.5 |
| 5-fold CV without augmentation | 64.4 ± 5.1 (same settings), Stage 3: 63.6 ± 4.0 | 60.9 ± 4.0 (Stage 4) |
| **5-fold CV with augmentation** | **66.0 ± 4.7** (folds 66.7, 65.2, 59.2, 73.9, 65.1) | **62.8 ± 3.6** (folds 65.8, 60.1, 58.5, 68.2, 61.7) |
| Paired gain from augmentation | +1.6 vs same-settings run (better in 3 of 5 folds, std error 1.1); +2.4 vs Stage 3 (4 of 5, std error 1.2) | +1.9 vs Stage 4 (5 of 5 folds, std error 0.4) |
| Leakage gap without / with augmentation | +4.8 / +5.9 (unchanged within noise) | +7.6 / +3.9 (seen 56.6, unseen 52.7 with augmentation) |

Augmented CNN is better than the augmented GRU in 5 of 5 folds (mean +3.2, std error 1.0) and 4 to 5 times faster (about 50 s against 180 to 310 s per fold).

**Ensembles.** Single test fold: 3 augmented CNNs 71.1 % (+3.8 over a single model), 3 augmented GRUs 71.9 % (+6.2), all 6 models 73.6 % (top-5 94.1 %). The same ensembles gain only +1.3, +0.1 and +1.8 on the validation fold, so test-fold gains are inflated by an easier fold.
**5-fold CV of one CNN + one GRU per fold: 67.7 ± 3.7 %** (+4.1 over the Stage 3 CNN in 5 of 5 folds, std error 0.8; +1.7 over the augmented CNN in 4 of 5 folds, std error 0.7). Not tested: whether two CNNs with different seeds give a similar gain (architectural diversity versus plain averaging).

**Possible data issue (checked, small but heavy tail).** The diagnostic plot (one clip chosen for its large fingertip movement) showed normalized fingertip x values of 80 to 90 hand units. Measured on the training clips (visible hands): median absolute normalized value 0.62, 90th percentile 1.69, 99th 4.25, 99.9th 14.30; 0.72 % of values above 5 and 0.19 % above 10. So the bulk is well behaved and the tail is small but heavy. Stage 4c (notebook above) tests whether repairing it helps. The per-frame normalization divides by the wrist to middle-finger-base distance in the image plane, which can become tiny when a hand points toward the camera. Not quantified yet: check the distribution of absolute normalized values (percentiles, share above 5 and above 10) and consider a more robust hand-size scale (for example including z, or a palm width) and/or clipping.

---

## 10c. Stage 4c: robust hand-size normalization

**Notebook:** `stage4c_robust_normalization.ipynb` (CNN + rotation recipe, about 60 to 73 s per run on a T4; 13 minutes for the 3-seed part, about 25 minutes for the 5-fold CV of the four variants).

**Diagnosis.** On the training clips (visible hands) the absolute normalized values had median 0.62, 90th percentile 1.69, 99th 4.25, 99.9th 14.30, maximum 293.75; 0.72 % above 5, 0.19 % above 10. The per-frame scale (wrist to middle-finger base in the image plane) has percentiles 1/5/50/95 of 0.0144/0.028/0.060/0.1246 image heights, and **the 5 % of frames with the smallest scale contain 100 % of the frames that have a value above 10**.

**Variants** (same CNN, rotation recipe, folds, settings): `base` (current), `clip5` (limit values to +/-5), `robust3d` (scale = mean 3D length of the four palm bones wrist to index/middle/ring/pinky finger base, z multiplied by the aspect ratio like x), `robust3d_clip5`. The tail statistics become: robust3d median 0.48, 99.9th percentile 2.23, maximum 5.29.

| Variant | Val top-1 (3 seeds) | Test top-1 (3 seeds) | Test top-5 | 5-fold CV top-1 | Paired vs base |
|---|---|---|---|---|---|
| base | 65.9 ± 0.8 | 65.2 ± 1.8 | 91.9 | 64.5 ± 4.5 (66.7, 64.9, 57.7, 71.1, 62.0) | n/a |
| clip5 | 65.7 ± 0.7 | 68.5 ± 0.4 | 92.3 | 65.0 ± 3.1 | +0.5 (2 of 5 folds, std error 0.8) |
| **robust3d** | **68.6 ± 1.3** | **71.5 ± 0.1** | 92.8 | **68.6 ± 3.9** (72.4, 65.2, 64.3, 73.9, 67.0) | **+4.0 (5 of 5, std error 1.1)** |
| robust3d_clip5 | 68.1 ± 0.6 | 70.9 ± 0.6 | 93.3 | 68.7 ± 3.6 | +4.2 (5 of 5, std error 0.6) |

The same configuration (CNN + rotation, base normalization) gave 66.0 ± 4.7 in Stage 4b versus 64.5 ± 4.5 here. **Correction (see section 10d):** a repeat of the robust3d run in the final pipeline gave 66.3 instead of 68.6, so the run-to-run noise of a single-seed CV is about 2 points and the gain from the robust scale is better estimated at about +2 points (range +1.5 to +3), not +3 to +4.

**Conclusions.** The problem was the collapsing single-bone scale; clipping alone does not fix it (the whole frame is mis-scaled, not only the extreme values); the robust scale was better in every fold of this experiment (but see the correction above). Not separated yet: gain from averaging four bones versus gain from using z (z unit assumed equal to x). All earlier stages used the old normalization; the GRU and the ensembles were re-run with `robust3d` in the final pipeline (section 10d).

---

## 10d. Final pipeline: robust normalization, repeated runs and ensembles

**Notebook:** `final_pipeline_robust_ensembles.ipynb` (robust3d + rotation augmentation; CNN runs 31 to 76 s, GRU CV runs 161 to 277 s; the GRU part took 40 minutes). It saves `results/final_cv_logits.npz` (cross-validated logits of every model, with clip ids and labels per fold) for the error analysis.

| Model | Val top-1 (3 seeds) | Test top-1 (3 seeds) | 5-fold CV top-1 |
|---|---|---|---|
| CNN robust3d + rotation | 67.8 ± 1.0 | 71.1 ± 1.5 (top-5 93.2) | seed 0: 66.3 ± 2.0 (68.7, 64.7, 63.5, 68.2, 66.2); seed 1: 66.6 ± 4.1 (72.9, 65.2, 60.8, 69.2, 64.8); mean of the two seeds per fold 66.4 ± 3.0 |
| GRU robust3d + rotation | 64.6 ± 0.8 | 67.8 ± 1.2 | 64.0 ± 4.1 (69.2, 60.9, 58.5, 68.2, 63.4) |
| Ensemble: two CNN seeds | n/a | n/a | 68.6 ± 3.8 (73.8, 67.1, 63.0, 71.9, 67.3) |
| Ensemble: one CNN + one GRU | n/a | n/a | 67.9 ± 4.4 |
| **Ensemble: two CNN seeds + one GRU** | n/a | n/a | **69.7 ± 4.1** (75.1, 67.4, 63.5, 73.4, 69.3) |

Single test fold seed ensembles: 3 CNNs 75.1 % (top-5 94.9, +4.0 over a single model; +2.4 on validation), 3 GRUs 71.6 % (+3.8; +1.4 on validation), all 6 models 74.3 % (+4.9; +2.6 on validation).

**Findings.**
1. **Run-to-run noise is about 2 points.** Same configuration and seed: 68.6 in Stage 4c, 66.3 here; seed 1: 66.6; per-fold differences between seeds up to 4 points. Any single-run CV difference below 2 to 3 points is not trustworthy; average several seeds before claiming anything.
2. **Robust normalization gain is about +2 points** (robust CV runs 68.6, 66.3, 66.6, mean 67.2, against old-normalization runs 66.0, 64.5, mean 65.3; validation +2.3). The earlier +4 estimate was inflated by a lucky run. Final CNN vs Stage 3 CNN: +2.6 points, 5 of 5 folds (standard error 1.2).
3. **GRU:** 64.0 in CV (+3.1 vs Stage 4 GRU in 5 of 5 folds, +1.2 vs Stage 4b GRU); the CNN is better in 4 of 5 folds (mean 2.2 points).
4. **Ensembles:** averaging works (two CNN seeds +2.4 over one in CV, 4 of 5 folds). CNN + GRU (67.9) is not better than two CNN seeds (68.6): no evidence that architectural diversity matters beyond averaging (confounded because the GRU is weaker). Adding the GRU to two CNNs gives +1.1 in all 5 folds (standard error 0.3). A third CNN seed was not tested. Best CV result: 69.7 ± 4.1 (+6.1 over the Stage 3 CNN).
5. **Leakage:** CNN seen 66.4 ± 0.7 vs unseen 59.8 ± 1.2 (gap +6.6, 54 fixed epochs); GRU seen 62.7 ± 0.1 vs unseen 57.2 ± 0.2 (gap +5.6, 85 epochs). Unseen-signer accuracy improved vs the earlier runs (CNN 57.3 to 59.8, GRU+rotation 52.7 to 57.2) but gaps did not shrink. Neural-network gaps over all runs lie between +3.9 and +7.6 (standard error about 2.7 each), against +0.5 for logistic regression and +2.9 for the MLP.

---

## 10e. Stage 7a: error analysis (no training)

**Notebook:** `stage7a_error_analysis.ipynb` (CPU, about 1 to 2 minutes). Input: `results/final_cv_logits.npz` (CNN seeds 0 and 1, GRU seed 0, robust3d + rotation) and `splits.csv`. Each non-prolific clip is a test clip exactly once: 1,932 clips, 153 signers, 300 signs. Predictions below are from the average of the three models (69.8 %, identical to the final pipeline: per-fold 75.1, 67.4, 63.5, 73.4, 69.3). Outputs: `results/stage7a_clip_predictions.csv`, `stage7a_per_sign.csv`, `stage7a_confusion_pairs.csv`, `stage7a_summary.json`, figures `stage7a_*.png`.

| Quantity | Value |
|---|---|
| Ensemble accuracy, signer-cluster bootstrap 95 % interval | 69.8 % [67.0, 72.6] (CNN seed 0 66.3 [63.8, 68.6]; seed 1 66.7 [63.8, 69.3]; GRU 64.1 [61.2, 67.0]) |
| Top-k of the ensemble | k=1 69.8, k=2 82.9, k=3 88.1, k=5 92.1, k=10 95.6 |
| Per sign (median 5 test clips) | macro accuracy 69.5 %; 8 signs never recognized (седьмой, семь, те, смешной, восемь, довольно, любить, ты); 149 signs at 80 % or more |
| Clip length quartiles (frames) | 4-35: 61.0 %; 35-43: 67.7 %; 43-53: 69.9 %; 53-204: 81.4 % |
| Share of video frames missing | <2 %: 70.5; 2-5 %: 72.6; 5-10 %: 71.2; 10-20 %: 70.7; >20 %: 54.8 (168 clips); low coverage (>50 %): 23.5 % (34 clips) vs 70.7 % for the others |
| Hands visible in most frames | one hand: 61.6 % (674 clips, 44 % of the errors); two hands: 74.2 % (1,258 clips) |
| Signers with at least 15 test clips (50 signers, 77 % of clips) | accuracy 42.9 to 100 %, std 11.9 points vs about 8.4 expected from sampling noise alone |
| Confidence | under-confident but well ordered (accuracy 74 % at mean confidence 0.45, 97 % at 0.85). Answering only when confident: 90 % coverage 74.5 %, 80 % 78.6 %, 70 % 82.5 %, 60 % 85.9 %, 50 % 88.5 % |
| Model agreement (CNN seed 0 vs GRU) | same answer on 69 % of clips; ensemble 81.4 % right when they agree, 44.5 % when they disagree |

**Confusions.** 583 errors, 474 distinct wrong pairs, 67 pairs with at least 2 errors; only 27 % of repeated confusions are reciprocal. Most frequent: чудесный to прекрасный (7 of 12), те to это (5 of 12), прием пищи to еда (5 of 13), and 4 each for любить to любовный, смешной to смеяться, ты to тебе, Пока to пока (что), обеспокоен to беспокойство, вкус to язык, Свисток to застенчивый. Ten "sink" signs (тебе, довольство, язык, это, беспокойство, так себе, прекрасный, еда, застенчивый, любовный) attract 12.9 % of all errors (3.3 % expected by chance).

**Interpretation (hypotheses, not verified).**
1. The confusions are between grammatical variants of one word, near-synonyms, cardinal/ordinal numbers and pointing signs. They probably differ mainly in non-manual cues (mouthing, facial expression) or in pointing direction relative to the body, none of which hand landmarks carry. **To verify: watch the videos of the 10 most confused pairs.** If the hand movements are identical, the ceiling for hands-only input has been reached and face (mouth) landmarks are the logical next investment.
2. One-hand signs are 35 % of the clips and 44 % of the errors; if they matched two-hand accuracy the overall gain would be about 4 points.
3. Short clips are much worse, but this is confounded with the sign itself (simple signs are short). Cheap check: compare, within the same sign, shorter-than-median and longer-than-median clips.
4. Missing frames only matter above 20 % (8.7 % of clips, 13 % of errors): perfect handling would gain at most 1 to 2 points.
5. The signer effect is real beyond noise. Possible causes to test: handedness (a left-handed signer mirrors signs, and no mirroring augmentation was tested), camera set-up, hand size. A proxy: the share of one-hand clips whose hand is on the left or right half of the image, per signer, against accuracy.
6. Hubness: a few sink signs attract predictions. A cheap test on the saved logits: per-class normalization of the predicted probabilities (tuned on other folds).
7. Practical: top-3 accuracy is 88 %, and a confidence threshold gives 88.5 % accuracy at 50 % coverage; the agreement of two models is an even stronger uncertainty signal (81.4 % vs 44.5 %).

---

## 11. All results side by side, and what we learned

| Model | 5-fold CV top-1 | 3-seed test top-1 | Parameters | Leakage gap (seen - unseen) |
|---|---|---|---|---|
| Chance | 0.33 | 0.33 | n/a | n/a |
| kNN | 31.2 ± 3.9 | 31.3 (single) | n/a | -1.2 |
| PCA + logistic regression | 43.0 ± 4.2 | 43.3 (single) | n/a | +0.5 |
| Linear SVM | 46.8 ± 3.1 | 51.3 (single) | n/a | not measured |
| MLP on 648 features | 53.0 ± 3.7 | 57.3 ± 1.8 | about 0.54M | +2.9 |
| MLP on raw sequence | not run | 45.2 ± 0.5 | about 4.5M | not measured |
| **1D-CNN on raw sequence** | **63.6 ± 4.0** | **66.6 ± 1.3** | 0.75M | +4.4 |
| BiGRU + conv front-end | 60.9 ± 4.0 | 63.7 ± 0.9 | 2.17M | +7.6 |
| Ensemble of 3 GRUs | not run | 66.7 (single split) | 3 x 2.17M | not measured |
| 1D-CNN + rotation augmentation (4b) | 66.0 ± 4.7 | 67.3 ± 1.0 | 0.75M | +5.9 |
| BiGRU + rotation augmentation (4b) | 62.8 ± 3.6 | 65.7 ± 1.5 | 2.17M | +3.9 |
| **Ensemble: augmented CNN + augmented GRU, one of each per fold (4b)** | **67.7 ± 3.7** | not run | 0.75M + 2.17M | not measured |
| 1D-CNN + rotation + robust3d normalization (4c, one run) | 68.6 ± 3.9 | 71.5 ± 0.1 | 0.75M | not measured |
| **1D-CNN + rotation + robust3d (final pipeline)** | 66.3 ± 2.0 (seed 0), 66.6 ± 4.1 (seed 1) | 71.1 ± 1.5 | 0.75M | +6.6 |
| BiGRU + rotation + robust3d (final pipeline) | 64.0 ± 4.1 | 67.8 ± 1.2 | 2.17M | +5.6 |
| **Ensemble of two CNN seeds + one GRU (final pipeline)** | **69.7 ± 4.1** | not run | 2 x 0.75M + 2.17M | not measured |

What we learned:

1. **Structure matters most.** The same raw input gives 45 % with a flattened MLP and 66 % with a CNN. Sharing small filters along time is the single biggest lever so far.
2. **Learned filters beat hand-made summaries** by about 10 points.
3. **Hand shape is almost all the signal.** Position, motion summaries and visibility channels add nothing measurable (consistent from Stage 1 to Stage 3).
4. **Bigger is not better here:** with about 17 training clips per sign, larger CNNs gain nothing and the GRU gains little for a large cost.
5. **Overfitting is the central problem for every model** (training accuracy about 100 % vs validation 55 to 66 %). Regularization, early stopping and label smoothing help a little; seed averaging gives about +3 points.
6. **Signer leakage grows with model flexibility:** +0.5 (logistic regression), +2.9 (MLP), +4.4 (CNN), +7.6 (GRU). The GRU gap is about 2.8 standard errors; the others are individually inconclusive. The CNN is also more
   robust to unseen signers than the GRU (57.3 vs 50.1 on unseen signers). Caveats: section 6.4.
7. **Single-split numbers are optimistic** by 3 to 5 points relative to 5-fold CV.
8. **Augmentation helps only a little.** Only rotation had a clear positive sign (about +1.6 to +2.4 CNN, +1.9 GRU in CV, close to the noise level); value-perturbing augmentations (noise, wrist shift, repeated frames) slightly hurt. It did not shrink the CNN's leakage gap; it shrank the GRU's (+7.6 to +3.9), which is suggestive only.
9. **Ensembling is the larger gain:** CNN + GRU gives 67.7 % in CV, +4.1 over the Stage 3 CNN.
10. **CNN beats GRU after augmentation** in every fold and trains 4 to 5 times faster.
11. **Preprocessing mattered about as much as augmentation:** replacing the single-bone hand-size scale by a robust 3D four-bone scale gave about +2 points in CV (an early estimate of +4 was inflated by run-to-run noise), similar to rotation augmentation (+1.6 to +2.4). Clipping alone did not help.
12. **Single-run cross-validation is noisy (about 2 points).** Average several seeds before claiming a difference smaller than 3 points.
13. **Errors are concentrated:** between related words and pointing signs, in one-hand signs (61.6 % vs 74.2 % for two hands), in short clips and in clips with more than 20 % missing frames (section 10e). This points to missing non-manual information (face, body) more than to model or recipe issues.
14. **Averaging models is the most reliable remaining gain** (+2.4 for two CNN seeds; best CV result 69.7 % with three models); architectural diversity was not shown to add more than averaging.

---

## 12. What remains: the roadmap

### 12.1 Augmentation and ensembling: DONE (Stage 4b, results in section 10b)

The original plan is kept below for reference. Remaining ideas from it that were **not** done: learning-rate schedule, mixup on sequences, weight averaging, mirroring, a baseline of two CNN seeds in CV to separate architectural diversity from averaging.

**New immediate next steps (in order):** (1) DONE: error analysis (section 10e). (2) Stage 7b, cheap (CPU, saved predictions): within-sign effect of clip length, signer-level diagnostics (handedness proxy, aspect, mean missing rate), hubness correction of the class probabilities, calibration (temperature), error budget table. (3) **Watch the videos of the 10 most confused pairs** (human step: ask someone who knows the sign language, or inspect the hand movements) to decide whether face/mouth landmarks are needed. (4) If a handedness effect shows up: test mirror augmentation (swap hands and flip x). (5) Face (mouth) landmarks for the 300-sign subset if step 3 supports it (hours of CPU extraction; the videos must be downloaded). (6) Stage 5 Transformer; (7) final run on all 1,000 signs; (8) controlled leakage experiment; optionally re-run Stages 1 and 2 with the robust normalization for a consistent final table; (2) cheap follow-up on preprocessing: separate the effect of the four-bone average from the use of z, try a per-clip scale, a palm-width scale and a z-unit check; (3) Stage 5 Transformer with the rotation recipe and the robust normalization; (4) Stage 7 analysis; (5) optional body/face landmarks.

Rationale: every model overfits and part of it looks signer-specific, so more varied training data should help generalization to new signers. It must come **before** the Transformer, otherwise the Transformer
(data hungry) would be compared unfairly with a CNN that got augmentation.

Plan for a new notebook (use `stage3_cnn1d.ipynb` as the template, since the CNN is the best model):

- Implement augmentations on batches of normalized sequences (N, 64, 132), **only on visible hands and without breaking the visibility flags and the zero-fill of missing hands**:
  in-plane rotation of the joint xy coordinates, global scale jitter, speed change by time warping and re-sampling to 64 frames, Gaussian noise on joints, random dropping of frames or of single joints,
  wrist-position jitter, and as an experiment mirroring (swap hands and flip x; may change the meaning of some signs, so test it separately).
- Tune the strength of each augmentation on validation only (2 seeds), then each type alone, then the best combination.
- Retrain the CNN with the best recipe, run the 5-fold CV, and compare **per fold** with 63.6 %.
- Re-run the leakage test: does the seen-unseen gap shrink?
- Ensembles: several CNN seeds, and CNN + GRU (different architectures should make different errors). The notebooks currently keep logits only inside each run; save them for ensembling.
- Smaller recipe items (about 1 to 2 points expected): learning-rate schedule, mixup on sequences, weight averaging.

### 12.2 Stage 5: Transformer

Small Transformer encoder on the 64 x 132 sequence (linear projection, positional encoding, CLS token or pooling, 2 to 4 layers). Expect it to need the augmentation recipe to compete. Bar to beat: CNN 63.6 ± 4.0 %.

### 12.3 Stage 7: analysis and the final run

- Error analysis: confusion matrix, per-sign accuracy, signs that depend on face/body, effect of `low_coverage` clips.
- **Signer-scaling curve:** train with 25 %, 50 %, 100 % of the training signers.
- **Controlled leakage experiment:** add k clips from each test signer to the training set (k = 0, 1, 2, 5) and measure accuracy as a function of k. This isolates signer familiarity from signer difficulty, which the official-split test cannot.
- Bootstrap confidence intervals (resampling clips, ideally signers) for the leakage gaps.
- Leave-one-prolific-signer-out evaluation (each of the 7 has hundreds of clips).
- Final comparison on **all 1,000 signs** with the same split (the signer assignment already covers them).
- Optional: body and face landmarks for the subset (MediaPipe Holistic on CPU, hours; many signs use the face and body), a real-time webcam demo (MediaPipe hands map to the 21-point layout; body points would need mapping).

### 12.4 Report

The report (Claude Doc) covers sections up to Stage 4. To do: add Stages 5 to 7, finish the references (below), add figures, and rewrite the introduction and abstract for the final results.

### 12.5 Open housekeeping

- Reference [5] (IEEE 10670380) is unidentified; references [6], [7], [9], [10] are listed by title only (authors missing). Ask the owner for the third article's title.
- Check the Slovo license terms (CC BY-SA variant) before publishing code or results.

---

## 13. Mistakes and pitfalls (read this before touching anything)

1. **An assumption about the official split was wrong.** The dataset description suggested the official split separates signers; it does not. It was caught by an assertion in Stage 0. Lesson: verify claims about data with code, and keep such assertions.
2. **Running cells from different notebook versions** (or skipping cell 1 after a restart) caused several errors (`NameError: N_FOLDS`, stale split cell, a missing column). Always open the latest notebook in a fresh session and run it top to bottom.
3. **Running cells out of order corrupted diagnostics:** the raw frame array was overwritten by the tracked one, so a "before tracking" statistic was computed on tracked data. The notebook now keeps `frames_raw` and `frames_tracked` separate and is safe to re-run.
4. **Random signer splits are unsafe** because 7 signers hold 68 % of the clips (section 3.3). Do not create new splits; use `splits.csv`.
5. **The 100-sign subset was too small:** about 130 clips per validation or test fold and many signs with no clip. That is why the subset has 300 signs.
6. **The linear SVM is slow** (about 10 minutes per fit; the 5-fold run took more than an hour). It is switched off in cross-validation by default.
7. **Single test folds mislead** (optimistic by 3 to 5 points). Always quote 5-fold results.
8. **top-5 for kNN with k = 1 is meaningless** (only one class is scored).
9. **GRU/CNN input layouts differ:** CNN expects (batch, channels, time), GRU (batch, time, channels). Passing the wrong dimension to the model builder was a real bug caught by testing.
10. **How the code was tested:** the assistant that wrote the notebooks had no PyTorch and no internet in its sandbox. The data handling, experiment loops, cross-validation and saving were tested end to end on synthetic data with a numpy stand-in for the models;
    the PyTorch parts (model classes and training loop) were reviewed by hand and then **executed for the first time on the owner's Colab, where they worked**. Treat the PyTorch code as validated by those runs.
11. **Normalizing by one projected bone is fragile:** in 5 % of frames the scale collapses (hand pointing at the camera, small hand), producing values up to 294 hand units. Use the robust 3D scale from Stage 4c (`normalize_hands_v(..., 'robust3d')`) for any new work.
12. **A single cross-validation run can mislead by about 2 points.** The robust-normalization gain was first estimated at +4 points from one run and later corrected to about +2 after a repeat. Repeat important comparisons with several seeds.
13. **Colab limits:** GPU quota and idle disconnects. Every notebook caches expensive results on Drive and checkpoints nothing else; re-running after a disconnect repeats the finished work of the current cell.

---

## 14. Limitations and open questions

- **Hands only.** Many signs use the face and body; the provided landmarks do not include them. This probably caps accuracy.
- **Few clips per sign** (about 17 for training in the subset) and **extreme signer imbalance** (7 signers = 68 % of the data), so the model is mostly trained on a handful of people and tested on the "tail" signers.
- **Selection bias, small and shared by all stages:** the hyperparameters were chosen on validation fold 1, which is also the test fold of cross-validation fold 1.
- **Greedy tuning:** the sweeps change one design choice at a time in a fixed order, with 2 seeds per option; differences between options are often within noise.
- **One seed per CV fold**, so seed variance is not included in the CV standard deviation.
- **The leakage comparison is confounded** by signer type (section 6.4) and by training-set size (the leakage runs train on 4,500 clips, the main runs on 5,222).
- **Russian Sign Language only.** Findings concern the method; transfer to another language (for example Arabic, relevant for the owner's region) is untested. Arabic datasets were found hard to obtain.
- Temporal distortion from resampling frames that are not evenly spaced (about 6.7 % of frames missing).
- Open scientific questions: does augmentation reduce the leakage gap? Does a Transformer help or just overfit? How much would face and body landmarks add? How many signers does a dataset need?

---

## 15. References

Complete citations are in the report; this list is for orientation.

1. T. Tao, Y. Zhao, T. Liu, J. Zhu, "Sign Language Recognition: A Comprehensive Review of Traditional and Deep Learning Approaches, Datasets, and Challenges", IEEE Access 12, 75034-75060, 2024. doi:10.1109/ACCESS.2024.3398806 (IEEE 10526274).
2. Y. Zhang, X. Jiang, "Recent Advances on Deep Learning for Sign Language Recognition", CMES 139(3), 2399-2450, 2024. doi:10.32604/cmes.2023.045731. Key points used: fingerspelling accuracy is high because vocabularies are small; accuracy collapses at larger vocabularies
   (one pose-based system 80.9 % on WLASL100 vs 38.65 % on WLASL2000); open problems: signer independence, no standard evaluation protocol, neglect of non-manual cues, lightweight models.
3. "A Comprehensive Review of Deep Learning Approaches for Video-Based Sign Language Recognition: Datasets, Challenges and Insights", Multimodal Technologies and Interaction, 2026. doi:10.3390/mti10060058.
4. A. Kapitanov, K. Kvanchiani, A. Nagaev, E. Petrova, "Slovo: Russian Sign Language Dataset", ICCVS 2023, arXiv:2305.14527.
5. IEEE Xplore 10670380: **not identified**.
6. "Multi-cue temporal modeling for skeleton-based sign language recognition" (PMC10113557).
7. "Part-Wise Graph Fourier Learning for Skeleton-Based Continuous Sign Language Recognition", J. Imaging 2025 (PMC12387829).
8. P. M. Ferreira, D. Pernes, A. Rebelo, J. S. Cardoso, "Learning signer-invariant representations with adversarial training", Proc. SPIE 11433, 2020.
9. "Signer-independent sign language recognition with feature disentanglement", Turkish J. Electrical Engineering and Computer Sciences 32(3).
10. "Improving Continuous Sign Language Recognition with Consistency Constraints and Signer Removal", arXiv:2212.13023.
11. R. Haque, M. Islam, S. M. T. U. Raju, F. Karray, "A Signer-Invariant Conformer and Multi-Scale Fusion Transformer for Continuous Sign Language Recognition", arXiv:2508.09372, 2025.

Related work on signer invariance (items 8 to 11) shows the idea of signer-invariant learning is established, on images, on other datasets, or for continuous recognition. Our contribution is the controlled, reproducible measurement on Slovo skeleton data, not the idea itself.

---

## 16. Glossary

- **Landmark:** a point on the body located by a program in an image (21 per hand), with x, y, z.
- **Clip / frame:** one video of one sign / one image of the video.
- **Signer-disjoint:** no person appears on more than one side of the split.
- **Signer leakage:** the model benefits from having seen the same person in training (their style), which inflates accuracy.
- **Prolific signer:** one of the 7 people who recorded most of the clips; always in training.
- **Fold:** one of 5 groups of signers with roughly equal clip counts; each is the test set once in cross-validation.
- **Top-1 / top-5:** the right sign is the first guess / among the five best guesses.
- **Overfitting:** the model memorizes training clips (about 100 % training accuracy) but does worse on new ones.
- **Regularization:** dropout, weight decay, label smoothing, early stopping: ways to reduce overfitting.
- **Epoch / early stopping:** one pass over the training data / stop when validation accuracy stops improving and restore the best weights.
- **Seed:** the starting number of the random generator; the same seed reproduces the same random choices.
- **Receptive field:** how many input frames influence one unit of a convolutional network.
- **Ensemble:** averaging the predictions of several models.
- **Paired comparison:** comparing two models on the same folds and the same test clips.
