# Particle Phase Classification in Mixed-Phase Clouds

Automated classification of liquid and ice particles from airborne 2D-S probe imagery using hybrid convolutional neural networks.

---

## Overview

The **NSF/NCAR Research Aviation Facility** uses the [Two-Dimensional Stereo Particle Imaging Probe (2D-S)](https://www.eol.ucar.edu/instruments/two-dimensional-stereo-particle-imaging-probe) to capture binary shadowgraph images of cloud particles during research flights. Determining whether a particle is **liquid** (supercooled water droplet) or **solid** (ice crystal) is straightforward at temperature extremes but difficult in the **mixed-phase range** (−40 °C to +1 °C), where both phases coexist and manual inspection is the only current method.

This project trains a hybrid CNN — combining particle images with a morphological shape feature — to classify particles as liquid or solid, and provides a ready-to-run inference pipeline for applying the model to new flight data.

**Key results (CGWaveS RF02 dataset):**

- Hybrid CNN (image + arearatio) outperforms image-only and feature-only baselines
- Validated with 5-fold cross-validation and 5 random seeds for statistical significance
- A separate lightweight CNN is provided to filter out donut artifacts before phase classification

---

## Labeling Strategy

| Temperature | Label |
| --- | --- |
| ATX ≥ 1 °C | Liquid (phase = 0) |
| ATX ≤ −40 °C | Solid (phase = 1) |
| Between −40 °C and 1 °C | Mixed-phase — model target |

Labels are derived from the concurrent environmental netCDF (ATX variable). Particles in the mixed-phase range have no temperature-based ground truth and are the primary inference target.

---

## Repository Structure

```text
aircraft_ML/
│
├── README.md                               ← this file
├── PROCESSING_README.md                    preprocessing notes and data details
│
├── ── Preprocessing ─────────────────────────────────────────────────
│
├── process_particle_data.py                MAIN preprocessing script
│                                           raw PBP netCDF → 128×128 PNG images
│                                           + particle_df.csv
├── synthetic_preprocessing.py              preprocessing script for synthetic particles
│
├── ── Model Training ────────────────────────────────────────────────
│
├── particle_classification_ablation_study.ipynb  ★ PRIMARY training notebook
│                                           ablation study: image-only vs.
│                                           features-only vs. hybrid CNN
│                                           5-fold CV + 5 seeds + significance tests
│                                           saves → ablation_results/*.keras
│                                                    data_splits_ablation/*.pkl
│
├── donut_filter_training.ipynb             ★ Donut filter training notebook
│                                           binary CNN: donut vs. not-donut
│                                           saves → donut_filter_results/
│                                                    donut_filter_model.keras
│
├── ── Inference ─────────────────────────────────────────────────────
│
├── particle_phase_inference.ipynb          ★ Apply trained models to new flight data
│                                           loads new PBP netCDF → quality filter
│                                           → (optional) donut filter
│                                           → phase classifier
│                                           → CSV + time-series plots
│
├── ── Historical / Reference ────────────────────────────────────────
│
├── particle_classification_CNN.ipynb       earlier 4-class hybrid CNN iteration
├── particle_classification_CNN_image.ipynb earlier image-only 4-class CNN
│
├── old_files/                              archived scripts and notebooks
│                                           (no longer part of active pipeline)
│
└── ── Saved Artifacts (generated on first run) ──────────────────────
    ├── ablation_results/                   trained phase models (*.keras)
    │   ├── hybrid_seed42.keras             ← default model used by inference
    │   ├── image_only_seed42.keras
    │   └── features_only_seed42.keras
    ├── data_splits_ablation/
    │   └── data_splits_seed42.pkl          ← train/val/test splits + scaler
    └── donut_filter_results/
        └── donut_filter_model.keras        ← donut filter model
```

---

## How to Run

### Step 1 — Preprocess a new flight (skip if using existing CGWaveS data)

Edit the file paths at the top of `process_particle_data.py`, then run:

```bash
python process_particle_data.py
```

This reads raw PBP netCDF files, applies quality filters, extracts 128×128 PNG images into `particle_images_filtered/{liquid,solid,donut,noise}/`, and writes `particle_df.csv`.

### Step 2 — Train the phase classifier

Open and run **`particle_classification_ablation_study.ipynb`** top to bottom.

- Trains three model variants (hybrid, image-only, features-only) across 5 seeds
- Performs statistical significance testing between models
- Saves the best model per variant to `ablation_results/`
- Saves the fitted `StandardScaler` inside `data_splits_ablation/data_splits_seed42.pkl`

### Step 3 — Train the donut filter (optional but recommended)

Open and run **`donut_filter_training.ipynb`** top to bottom.

- Trains a binary CNN (donut vs. not-donut) on the labeled image set
- Produces a threshold sensitivity plot to help choose `DONUT_THRESHOLD`
- Saves to `donut_filter_results/donut_filter_model.keras`

### Step 4 — Apply to new flight data

Open **`particle_phase_inference.ipynb`** and edit the configuration cell:

```python
NEW_PBP_FILES    = ['Data/new_flight_F2DS_V.pbp.nc', ...]
FLIGHT_DATE      = 'YYYY-MM-DD'
MODEL_PATH       = './ablation_results/hybrid_seed42.keras'
DATA_SPLITS_PATH = './data_splits_ablation/data_splits_seed42.pkl'

DONUT_FILTER     = True   # recommended
DONUT_MODEL_PATH = './donut_filter_results/donut_filter_model.keras'
DONUT_THRESHOLD  = 0.5    # tune from threshold_sensitivity.png
```

Run the notebook top to bottom. Outputs land in `inference_results/`.

---

## Model Architecture

### Phase Classifier — Hybrid CNN

The primary model takes two inputs and predicts Liquid (0) vs. Solid (1):

```text
Image branch                     Feature branch
128×128×1 grayscale              [arearatio]  ← area / bounding-box area
    │                                │
Conv2D(32) → BN → Pool → Drop   Dense(16) → ReLU → Drop(0.2)
Conv2D(64) → BN → Pool → Drop       │
Conv2D(128) → BN → Pool → Drop      │
Conv2D(128) → BN → Pool → Drop      │
Flatten → Dense(128) ────────────────┤
                              Concatenate (160-dim)
                                    │
                     Dense(256, ReLU, L2) → Drop(0.3)
                     Dense(128, ReLU)     → Drop(0.5)
                     Dense(64,  ReLU)     → Drop(0.3)
                     Dense(2,   Softmax)
```

**Feature used:** `arearatio` — the ratio of particle area to its filled bounding-box area. Chosen after testing for feature leakage; captures shape compactness without introducing temperature signal.

### Donut Filter — Image-Only CNN

Lightweight binary CNN (P(donut) ∈ [0, 1]):

```text
128×128×1 → Conv2D(32) → BN → Pool → Drop
          → Conv2D(64) → BN → Pool → Drop
          → Conv2D(128) → BN → Pool → Drop
          → Flatten → Dense(64) → Drop → Dense(1, Sigmoid)
```

Runs before the phase classifier. Particles with P(donut) ≥ `DONUT_THRESHOLD` are labelled `Donut` (phase = 2) and excluded from phase classification.

---

## Dataset

| Property | Value |
| --- | --- |
| **Source** | CGWaveS RF02, 2025-05-24, NCAR HIAPER (NSF/NCAR) |
| **Probes** | F2DS vertical + horizontal |
| **Total particles after filtering** | 7,182 |
| **Liquid (phase 0)** | 2,302 |
| **Solid (phase 1)** | 1,950 |
| **Donut (phase 2)** | 2,230 |
| **Noise (phase 3)** | 700 |
| **Binary labeled set (liquid + solid)** | 4,252 |
| **Image format** | 128×128 px grayscale PNG, white background |
| **Scalar feature** | `arearatio` (1 feature) |

---

## Quality Filters

Applied identically during preprocessing and inference. A particle is removed if any of the following hold:

| Filter | Criterion |
| --- | --- |
| Too small | diameter ≤ 100 µm |
| Hollow / donut-like | void index ≥ 0.05 (heuristic) |
| Near-perfect rectangle | arearatiofilled > 0.95 AND aspectratio > 0.90 |
| Line-like | aspectratio < 0.20 |
| Long rectangle | arearatiofilled ≥ 0.90 AND aspectratio < 0.15 |

The CNN donut filter (`DONUT_FILTER = True`) provides an additional learned check on top of these heuristics.

---

## Inference Output

`inference_results/particle_phase_predictions.csv` — one row per particle:

| Column | Description |
| --- | --- |
| `particle_id` | sequential ID from PBP file |
| `probe` | V or H |
| `time` | UTC timestamp |
| `arearatio` | raw shape feature |
| `predicted_phase` | 0 = Liquid, 1 = Solid, 2 = Donut |
| `predicted_label` | human-readable label |
| `prob_liquid` | P(Liquid) from phase classifier |
| `prob_solid` | P(Solid) from phase classifier |
| `confidence` | max probability (phase confidence, or donut_prob for flagged particles) |
| `is_donut` | bool — only present when `DONUT_FILTER = True` |
| `donut_prob` | P(donut) — only present when `DONUT_FILTER = True` |

---
