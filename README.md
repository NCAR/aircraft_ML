# Particle Phase Classification in Mixed-Phase Clouds

## Proposal Title
**Classification of Liquid and Ice Particles in Mixed-Phase Clouds via Hybrid Convolutional Neural Network (CNN) and Multimodal Feature Learning**

---

## 1. Problem Significance

**Cloud composition is critical in climate modeling and weather prediction**, as the phase of water (liquid, solid, or mixed) governs fundamental physical processes like solar radiation transfer (albedo), energy exchange, and precipitation formation.

The **NSF/NCAR Research Aviation Facility** utilizes airborne instruments, such as the **Two-Dimensional, Stereo, Particle Imaging Probe (2D-S)**, which captures a **binary 2D image** (shadowgraph) of particles. This provides high-resolution data on size and complex shape.

Cloud composition is critical in climate modeling and weather prediction because the phase of water governs physical processes like albedo, energy exchange, and precipitation formation. The NSF/NCAR Research Aviation Facility uses airborne instruments, like the [Two-Dimensional, Stereo, Particle Imaging Probe (2D-S)](https://www.eol.ucar.edu/instruments/two-dimensional-stereo-particle-imaging-probe) to study these cloud microphysics. The probe captures a binary 2D image representation as the particles pass through the probe, to provide high-resolution data on size and complex shape.

* **Definitive Phases:** In air temperatures above 0-1 °C, particles are assumed liquid (water). Below -40 °C, they are assumed solid (ice).
* **Mixed-Phase:** Between these values is the **mixed-phase range**, where supercooled water and ice particles coexist, and we are unable to assume the particle state based on air temperature.
* **Current Challenge:** Currently, phase classification within the mixed-phase range is tedious, often relying on manual inspection of the imagery. There is no reliable, automated way to accurately classify these particles.

I am seeking to address this gap by using a CNN on the particle-by-particle data to classify mixed-phase cloud particles as either liquid or ice. I will first train just using the particle images, and then I will add in environmental context to help refine the model.

---

## 2. Machine Learning Task Description

The core objective is a **Binary Classification** task to identify if a particle is liquid or solid.

### A. Data Integration and Preprocessing

1. **Data Filtering:** To make sure our data is clean for training, we are filtering out particles under 100 microns and perfectly rectangular particles (with a 5% margin). This removes small particles that show up as square and could confuse the model, and removes rectangles which are almost always noise.
2. **Labeling:** Using the temperature data taken concurrently alongside the particle measurements, our labeled dataset is created.
3. **Image Segmentation & Standardization:** The NetCDF file contains the raw imagery data and the boundary indices ($\mathbf{starty}$, $\mathbf{stopy}$, $\mathbf{startx}$, $\mathbf{stopx}$) for each particle. I am extracting cropped 2D binary images for each particle using these indices. Each image is then scaled (preserving its aspect ratio) and zero-padded to a uniform **$128 \times 128$ pixel** canvas size, creating the required grid input for the CNN.
4. **Class Imbalance:** I have ~3.6 times the images for solid as I do for liquid. To handle the class imbalance, I will try oversampling the liquid images with augmentation, ensuring balanced batches.

### B. Classification Model (CNN --> Hybrid CNN)

Initially, I will test a CNN just using the labeled images to see how it does, first with randomly initialized weights, and then exploring options for pretrained weights.

I ultimately expect to employ a **Multi-Modal Convolutional Neural Network (CNN)** to process both data types as referenced [here](https://rosenfelder.ai/multi-input-neural-network-pytorch/):

1. **Image Branch (CNN):** The $\mathbf{128 \times 128 \times 1}$ grayscale image is processed through 4 convolutional blocks:
    * Conv2D layers with filters: 32 → 64 → 128 → 128
    * Each block includes BatchNormalization and MaxPooling (2×2)
    * Output is flattened and passed through Dense layers (256 neurons) with Dropout (0.3)
2. **Tabular Branch (Fully Connected or MLP):** The features (temperature, air speed, altitude) are processed through:
    * Dense layers: 32 → 16 neurons
    * Dropout regularization
3. **Fusion & Prediction:** Feature vectors from both branches are concatenated and passed through additional Dense layers (128 → 64 neurons) with Dropout ending with a **softmax output layer** producing binary class probabilities (0: Liquid, 1: Ice).

## 3. Dataset Characteristics

| Metric | Detail |
| :--- | :--- |
| **Project Data Source** | CGWaveS RF02 project data (with potential for multi-project validation) |
| **Dataset Size (Labeled Samples)** | $\mathbf{25,270}$ individual particles after filtering |
| **Target Variable** | **Particle Phase (Binary Classification):** 0 (Water) / 1 (Ice) |
| **Input Feature Type** | **Multimodal** (Image Array + Tabular Vector) |
| **Image Input Dimensions** | $128 \times 128 \times 1$ (Standardized Grayscale Image) |
| **Tabular Features** | 3 (Temperature, Air Speed, Altitude) |

---

## Appendix A: Implementation Status

| Component | Status | Details |
| :--- | :--- | :--- |
| **Data Access** | Complete | NetCDF particle-by-particle and state variables loaded in `pbp_plotting.ipynb` |
| **Pre-processing** | Complete | Image extraction, scaling, and standardization to $128 \times 128$ implemented in `plot_particle_standardized()` function |
| **Image Export** | Complete | Standardized particle images saved. |
| **Model Architecture** | In progress | Multi-modal CNN implemented in `particle_classification_CNN.ipynb` with separate branches for images and environmental features |
| **Training Pipeline** | In progress | Complete training pipeline with data loading, splitting, callbacks, and evaluation. Train, validate, and refine the model on classified particles |

---

## Appendix B: Key Feature Variables

| Variable | Type | Role | Implementation |
| :--- | :--- | :--- | :--- |
| $\mathbf{image}$ | Array ($128 \times 128 \times 1$) | Primary input for CNN; provides detailed shape morphology | Processed through 4-block CNN architecture |
| $\mathbf{label}$ | Binary | Target variable: 0 (Liquid) / 1 (Ice) | One-hot encoded for training |

### Additional Variables Available (Not Currently Used)

* $\mathbf{aspectratio}$: Particle shape metric (could be added to tabular features)
* $\mathbf{diam}$: Particle size metric (could be added to tabular features)
* $\mathbf{area}$: Particle area (could be added to tabular features)
* $\mathbf{TASX}$ :True Air Speed for environmental context
* $\mathbf{GGALT}$ : Flight altitude for environmental context
* $\mathbf{ATX}$: Air temperature. Used for classification but could be used in the mix phase as a feature.
