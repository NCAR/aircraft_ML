# Particle Phase Classification in Mixed-Phase Clouds

## Proposal Title
**Classification of Liquid and Ice Particles in Mixed-Phase Clouds via Hybrid Convolutional Neural Network (CNN) and Multimodal Feature Learning**

---

## 1. Problem Significance

**Cloud composition is critical in climate modeling and weather prediction**, as the phase of water (liquid, solid, or mixed) governs fundamental physical processes like solar radiation transfer (albedo), energy exchange, and precipitation formation.

The **NSF/NCAR Research Aviation Facility** utilizes airborne instruments, such as the **Two-Dimensional, Stereo, Particle Imaging Probe (2D-S)**, which captures a **binary 2D image** (shadowgraph) of particles. This provides high-resolution data on size and complex shape.

* **Definitive Phases:** Above 0-1 °C, particles are assumed liquid (water). Below -40 °C, they are assumed solid (ice).
* **Mixed-Phase:** Between these values is the **mixed-phase range**, where supercooled water and ice particles coexist, and thus we are unable to assume the particle state.
* **Current Challenge:** Currently, phase classification within this range is tedious, often relying on **manual, subjective inspection of the imagery**. There is no reliable, high-throughput, automated way to accurately classify these particles.

This project addresses this gap by leveraging the power of **deep learning on the high-resolution imagery** combined with environmental context to classify mixed-phase cloud particles as either liquid or ice.

---

## 2. Machine Learning Task Description

The core objective is a **Binary Classification** task for particle-by-particle phase discrimination.

### A. Data Integration and Preprocessing

1.  **Image Segmentation & Standardization:** The NetCDF file contains the raw imagery data and the boundary indices ($\mathbf{starty}$, $\mathbf{stopy}$, $\mathbf{startx}$, $\mathbf{stopx}$) for each particle. We will use these indices to extract a minimally **cropped 2D binary image** for each particle. Each image will then be **scaled** (preserving its aspect ratio) and **zero-padded** to a uniform **$128 \times 128$ pixel** canvas size, creating the required grid input for the CNN.
2.  **Feature Fusion:** The standardized image is fused with its associated **scalar (tabular) features**. This includes particle metadata ($\mathbf{aspectratio}$, $\mathbf{diam}$, $\mathbf{area}$) and environmental state parameters ($\mathbf{Temperature}$, $\mathbf{Altitude}$, $\mathbf{True\ Air\ Speed}$) merged from the secondary NetCDF file.

### B. Classification Model (Hybrid CNN)

We will employ a **Hybrid Convolutional Neural Network (CNN)** architecture to simultaneously process both data types:

1.  **Image Branch (CNN):** The $\mathbf{64 \times 64}$ image is fed into a Convolutional Network. This branch automatically learns complex **morphological features** (e.g., jagged edges, internal structure) from the image shadow.
2.  **Tabular Branch (Fully Connected):** The scalar features are input into a smaller, fully-connected (Dense) network (e.g., temperature dependence).
3.  **Prediction:** The high-level feature vectors from both branches are **concatenated** and passed to a final classifier to produce the **binary prediction** (0: Water, 1: Ice).

### C. Training Strategy

The model will be trained on particles in the defined phases (e.g., $T > 0\ ^\circ\text{C}$ for Water and $T < -50\ ^\circ\text{C}$ for Ice). The task is to apply this trained classifier to predict the phase of particles within the **mixed-phase temperature range** (e.g., $-40\ ^\circ\text{C} \le T \le 0\ ^\circ\text{C}$).

---

## 3. Dataset Characteristics

| Metric | Detail |
| :--- | :--- |
| **Project Data Source** | CGWaveS RF02 project data (with potential for multi-project validation) |
| **Dataset Size (Samples)** | $\mathbf{3,754,562}$ individual particles |
| **Target Variable** | **Particle Phase (Binary Classification):** 0 (Water) / 1 (Ice) |
| **Input Feature Type** | **Multimodal** (Image Array + Tabular Vector) |
| **Image Input Dimensions** | $128 \times 128 \times 1$ (Standardized Image) |
| **Scalar Features** | $\approx 5+$ (Particle metrics + State parameters) |
| **Key Challenge** | **Class Imbalance** (requires careful tuning and weighted loss functions) |

---

## Appendix A: Proposal Status

| Component | Status | Next Step |
| :--- | :--- | :--- |
| **Data Access** | NetCDF particle-by-particle and state variables loaded (`dc`) | Merge state variables and particle data; generate ground truth labels. |
| **Pre-processing** | Code ready for image slicing and standardization (to $64 \times 64$). | Apply standardization function to all $3.75$ million images. |
| **Model Choice** | Hybrid CNN Architecture defined. | Implement and train the Hybrid CNN model. |

---

## Appendix B: Key Feature Variables

| Variable | Type | Role |
| :--- | :--- | :--- |
| $\mathbf{image}$ | Array (1281284) | Primary input for CNN; provides detailed shape morphology. |
| $\mathbf{ATX}$ | Continuous | Critical for labeling and contextual classification. |
| $\mathbf{aspectratio}$ | Continuous | Particle shape metric (Tabular input). |
| $\mathbf{diam}$ | Continuous | Particle size metric (Tabular input). |
| $\mathbf{TAS}$ | Continuous | True Air Speed (TAS) for environmental context ($\text{m/s}$). |
| $\mathbf{GGALT}$ | Continuous | GPS altitude ($\text{m}$). |
