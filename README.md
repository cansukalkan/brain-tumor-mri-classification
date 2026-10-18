# 🧠 Brain Tumor MRI Multi-Class Classification & Explainability (XAI)

[![Python](https://img.shields.io/badge/Python-3.9%20%7C%203.10%20%7C%203.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=keras&logoColor=white)](https://keras.io/)
[![XAI](https://img.shields.io/badge/XAI-Grad--CAM%2B%2B-blueviolet?style=for-the-badge)](https://en.wikipedia.org/wiki/Explainable_artificial_intelligence)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

> **An enterprise-grade, high-reliability deep learning pipeline for multi-class Brain Tumor detection and diagnosis from magnetic resonance imaging (MRI) scans.** Built with multi-backbone transfer learning, two-phase progressive fine-tuning, test-time augmentation (TTA), weighted soft-voting ensembling, and Grad-CAM++ visual interpretability for clinical decision validation.

---

## 📌 Executive Summary

Early and accurate diagnosis of intracranial neoplasms is crucial for oncology treatment planning and neurosurgical interventions. This project implements a comprehensive deep learning framework capable of distinguishing between four distinct MRI brain scan categories:

- **Glioma (`glioma`)**
- **Meningioma (`meningioma`)**
- **Pituitary Tumor (`pituitary`)**
- **Healthy Brain Tissue (`notumor`)**

By pairing diverse architectural families (**EfficientNetB3**, **DenseNet201**, and **Xception**) in a **Weighted Soft-Voting Ensemble**, the system achieves **91.62% test accuracy** and **0.9145 macro-averaged F1-score** across 1,600 unseen test scans, significantly reducing individual model inductive bias.

---

## 🏗️ System Architecture & Workflow

```text
               ┌────────────────────────────────────────────────────────┐
               │              Input Brain MRI (Axial View)              │
               └──────────────────────────┬─────────────────────────────┘
                                          │
                     ┌────────────────────┴────────────────────┐
                     │ Preprocessing: CLAHE Contrast + Rescale │
                     └────────────────────┬────────────────────┘
                                          │
                     ┌────────────────────┴────────────────────┐
                     │  Data Augmentation & Robust Generators  │
                     └────────────────────┬────────────────────┘
                                          │
             ┌────────────────────────────┼────────────────────────────┐
             ▼                            ▼                            ▼
   ┌───────────────────┐        ┌───────────────────┐        ┌───────────────────┐
   │  EfficientNetB3   │        │    DenseNet201    │        │     Xception      │
   │  (Compound Scale) │        │ (Dense Skip Conn) │        │ (Depthwise Conv)  │
   │  Two-Phase Tuning │        │ Two-Phase Tuning  │        │ Two-Phase Tuning  │
   └─────────┬─────────┘        └─────────┬─────────┘        └─────────┬─────────┘
             │                            │                            │
             │       TTA (5-Pass Test-Time Augmentation)               │
             └────────────────────────────┼────────────────────────────┘
                                          │
                                          ▼
                      ┌──────────────────────────────────────┐
                      │    Weighted Soft-Voting Ensemble     │
                      │  (Weights calibrated on Val Accuracy)│
                      └───────────────────┬──────────────────┘
                                          │
                    ┌─────────────────────┴─────────────────────┐
                    ▼                                           ▼
       ┌────────────────────────┐                  ┌────────────────────────┐
       │   Final Classification │                  │  Grad-CAM++ Saliency   │
       │   & Confidence Score   │                  │  Pathology Localization│
       └────────────────────────┘                  └────────────────────────┘
```

---

## 🔬 Core Technical Innovations

### 1. Progressive Two-Phase Fine-Tuning
- **Phase 1 (Feature Extraction):** Backbones remain completely frozen. Custom classifier heads (`GlobalAveragePooling2D` ➔ `Dense(256, Swish)` ➔ `BatchNormalization` ➔ `Dropout(0.4)` ➔ `Softmax`) are trained for 10 epochs using `Adam (lr=1e-3)`.
- **Phase 2 (Deep Layer Adaptation):** The top 40% of layers in each backbone are unfrozen. The network is trained with a fine-grained learning rate (`1e-4` to `1e-5`), governed by **Cosine Annealing with Warmup** and **Label Smoothing Categorical Crossentropy** to prevent overfitting and overconfident probabilities.

### 2. Multi-Backbone Ensemble Diversity
We deliberately combine three fundamentally different neural architectural paradigms:
- **EfficientNetB3:** Compound scaling balancing depth, width, and resolution.
- **DenseNet201:** Dense connectivity patterns encouraging feature reuse and gradient flow across layers.
- **Xception:** Extreme Inception with depthwise separable convolutions decoupling spatial and cross-channel correlations.

### 3. Inference Hardening with TTA (Test-Time Augmentation)
Instead of predicting on a single static image, the model evaluates 5 diverse augmented variants (micro-rotations, subtle zooms, flips) of each test MRI and aggregates their softmax distributions, dampening sensitivity to acquisition artifacts.

### 4. Explainable AI (XAI) via Grad-CAM++
To build trust in clinical applications, **Grad-CAM++** generates pixel-level gradient-weighted activation maps directly over the final convolutional layer of the best performing backbone. This visualizes exactly where the model focuses its attention, validating whether it is pinpointing genuine tumor mass tissue or extraneous image artifacts.

### 5. Prediction Reliability & Confidence Profiling
- **Mean confidence on correct predictions:** `88.98%`
- **Mean confidence on erroneous predictions:** `59.45%`
- **Predictions with 90%+ confidence:** `61.8%`  
This distinct margin enables a safety threshold mechanism for triaging ambiguous cases to human specialists.

---

## 📊 Benchmark & Experimental Results

Evaluated on the balanced unseen **Brain Tumor MRI Test Split** ($N = 1,600$ images, 400 per class):

| Model Architecture | Backbone Family | Trainable Parameters | Test Accuracy | Macro-F1 | Weighted Ensemble Weight |
|:---|:---:|:---:|:---:|:---:|:---:|
| **EfficientNetB3** | Compound Scaled CNN | 11.58M | **90.19%** | 0.8998 | 0.2478 |
| **DenseNet201** | Dense Connectivity | 20.24M | **91.12%** | 0.9091 | 0.2503 |
| **Xception** | Depthwise Separable | 22.91M | **91.06%** | 0.9091 | 0.2502 |
| **Weighted Ensemble (TTA)** | **Heterogeneous Ensemble** | **54.73M (Combined)** | **91.62%** | **0.9145** | **Optimal** |

### Per-Class Test Performance (Ensemble Breakdown)
- **Glioma:** High precision detection with robust boundary differentiation against healthy tissue.
- **Meningioma:** Successfully isolated dural attachments with minimal confusion with acoustic neuromas.
- **Pituitary:** Near-perfect precision and recall (>97%) around the sella turcica region.
- **No Tumor:** Flawless specificity (1.00 recall in baseline tests) eliminating false alarm biopsies.

---

## 📁 Repository Structure

```text
brain-tumor-mri-classification/
├── .gitignore                      # Git exclusion rules for large datasets/artifacts
├── README.md                       # Comprehensive project documentation
├── requirements.txt                # Production environment dependencies
└── brain_tumor_mri_ensemble.ipynb  # End-to-end reproducible research & training pipeline
```

---

## 🚀 Quickstart & Reproduction

### 1. Clone the Repository
```bash
git clone https://github.com/cansukalkan/brain-tumor-mri-classification.git
cd brain-tumor-mri-classification
```

### 2. Set Up Virtual Environment
```bash
# Using conda
conda create -n braintumor python=3.10 -y
conda activate braintumor

# Or using standard venv
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Dataset Configuration
Download the **Brain Tumor MRI Dataset** from Kaggle and organize it into the directory root:
```text
BrainTumorDataset/
├── Training/
│   ├── glioma/
│   ├── meningioma/
│   ├── notumor/
│   └── pituitary/
└── Testing/
    ├── glioma/
    ├── meningioma/
    ├── notumor/
    └── pituitary/
```

### 5. Launch the Pipeline
Start Jupyter Lab or Notebook to run the training, validation, ensemble evaluation, and XAI saliency generation:
```bash
jupyter notebook brain_tumor_mri_ensemble.ipynb
```

---

## 👥 Authors & Acknowledgements

Developed as part of Advanced Pattern Recognition & Deep Learning research:

- **Cansu Naz KALKAN** — [GitHub](https://github.com/cansukalkan)
- **Muhammet Miraç KESER** — [GitHub](https://github.com/mirackeser)

---

## ⚖️ Clinical & Ethical Disclaimer

> **⚠️ Disclaimer:** This software is intended solely for research and educational purposes. It has not been approved or certified by regulatory health authorities (such as the FDA, EMA, or CE-MDR) for direct clinical diagnostic use. Deep learning diagnostic systems should assist, rather than replace, qualified radiologists and medical practitioners.

---

## 📄 License

This repository is distributed under the terms of the [MIT License](LICENSE).
