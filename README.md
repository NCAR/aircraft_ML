# Particle Phase Classification in Mixed-Phase Clouds

Automated classification of liquid and ice particles from airborne 2D-S probe imagery using hybrid convolutional neural networks.

---

## Overview

The **NSF/NCAR Research Aviation Facility** uses the [Two-Dimensional Stereo Particle Imaging Probe (2D-S)](https://www.eol.ucar.edu/instruments/two-dimensional-stereo-particle-imaging-probe) to capture binary shadowgraph images of cloud particles during research flights. Determining whether a particle is **liquid** (supercooled water droplet) or **solid** (ice crystal) is straightforward at temperature extremes but difficult in the **mixed-phase range** (−40 °C to +1 °C), where both phases coexist and manual inspection is the only current method.

This project trains a hybrid CNN — combining particle images with morphological shape features — to classify particles as liquid or solid, and provides a ready-to-run inference pipeline for applying the model to new flight data.

**Key results (CGWaveS RF02 dataset):**

- Hybrid CNN (image + arearatio + aspectratio, gated fusion) outperforms image-only and feature-only baselines
- 93.3% accuracy, validated across 5 random seeds with statistical significance tests
- A separate lightweight CNN is provided to filter out donut artifacts before phase classification
- Synthetic particle data pipeline supported, with donut filter applied as pre-training cleanup

### Next Steps

Aaron shared some resources to help us incorporate into our model:

- CNN for holographic imaging probes. Used a method that 'fine-tuned' the model with a small fraction of the target data after pre-training on training data. This reduced the large data needs for the target: https://amt.copernicus.org/articles/13/2219/2020/
- [A Technique for Habit Classification of Cloud Particles](https://doi.org/10.1175/1520-0426(2000)017%3C1048:ATFHCO%3E2.0.CO;2)
- Using cloud statistics (instead of pbp statistics) as an input to the model to help with training. For example, in some mixed phase clouds, as they transition, the small particles will melt first, so the area ratio by particle size statistics might tell properties about the cloud. Might help with certain mixed phase scenarios if we have labeled training data. Maybe other grouped statistics can be incorporated.

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
│                                           5 seeds + significance tests
│                                           saves → ablation_results/*.keras
│                                                    data_splits_ablation/*.pkl
│
├── hybrid_model_synthetic.ipynb            ★ Train hybrid model on synthetic data
│                                           applies CNN donut filter before training
│                                           to remove mislabeled out-of-focus artifacts
│                                           saves → hybrid_synthetic_results/
│                                                    hybrid_synthetic_model.keras
│                                                    scaler.pkl
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
│                                           → writes predictions back to netCDF
│                                           includes standalone particle viewer
│                                           (section 11, no re-run required)
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
    ├── donut_filter_results/
    │   └── donut_filter_model.keras        ← donut filter model
    ├── hybrid_synthetic_results/           ← produced by hybrid_model_synthetic.ipynb
    │   ├── hybrid_synthetic_model.keras
    │   └── scaler.pkl
    └── inference_results/                  ← produced by particle_phase_inference.ipynb
        ├── particle_phase_predictions.csv
        ├── *_classified.nc                 ← original PBP files + phase predictions
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

### Step 2a (alternative) — Train on synthetic data

If using synthetic particle data instead of (or in addition to) real flight data, run **`hybrid_model_synthetic.ipynb`**:

- Loads `synthetic_particle_metadata.csv` and `synthetic_particle_images_filtered/`
- Applies the pre-trained CNN donut filter to remove mislabeled out-of-focus artifacts before training
- Trains the same hybrid architecture as the ablation study (2 classes: Liquid / Ice)
- Saves model and scaler to `hybrid_synthetic_results/`

Requires `donut_filter_training.ipynb` to have been run first.

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

Run the notebook top to bottom. Outputs land in `inference_results/`:
- `particle_phase_predictions.csv` — per-particle predictions
- `*_classified.nc` — original PBP netCDF files with three new variables added: `phase_predicted`, `phase_prob_liquid`, `phase_prob_solid`

To re-inspect particles after closing the notebook, run only the Configuration and Helper Functions cells, then run **Section 11 (Quick Particle Viewer)** — no need to re-run inference.

---

## Model Architecture

### Phase Classifier — Hybrid CNN

The primary model takes two inputs and predicts Liquid (0) vs. Solid (1).
Architecture from the ablation study (37,634 parameters):

```text
Image branch                        Feature branch
128×128×1 grayscale                 [arearatio, aspectratio]
    │                                       │
Conv2D(16) → BN → MaxPool(2×2)      Dense(128) → Drop(0.2)
Conv2D(32) → BN → MaxPool(4×4)      Dense(64)  → Drop(0.2)
GlobalAveragePooling2D              Dense(32)  → Drop(0.2)
Dense(32, ReLU) ─────────────────────────────┤
                                  Concatenate (64-dim)
                                        │
                           sigmoid gate (Multiply — gated fusion)
                                        │
                            Dense(128, ReLU) → Drop(0.2)
                            Dense(64,  ReLU) → Drop(0.2)
                            Dense(2,   Softmax)
```

**Features used:** `arearatio` (area / filled area) and `aspectratio` (minor / major axis). Both capture shape morphology without introducing temperature signal.

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
| **Scalar features** | `arearatio`, `aspectratio` (2 features) |

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

### CSV — `inference_results/particle_phase_predictions.csv`

One row per classified particle:

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

### NetCDF — `inference_results/*_classified.nc`

A copy of each input PBP file with three variables added along the `Time` dimension:

| Variable | Type | Description |
| --- | --- | --- |
| `phase_predicted` | int8 | 0 = Liquid, 1 = Solid, 2 = Donut, −1 = excluded by quality filter |
| `phase_prob_liquid` | float32 | P(Liquid); NaN for excluded or donut-flagged particles |
| `phase_prob_solid` | float32 | P(Solid); NaN for excluded or donut-flagged particles |

The `Time` dimension matches the original file exactly — unclassified particles receive fill values rather than being dropped.

---
