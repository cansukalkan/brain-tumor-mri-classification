# 🧠 Brain Tumor MRI Classification & Explainability (XAI)

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A multi-class deep learning pipeline designed to detect and classify brain tumors from magnetic resonance imaging (MRI) scans into four distinct categories:
- **Glioma**
- **Meningioma**
- **Pituitary**
- **No Tumor**

This project leverages transfer learning across multiple state-of-the-art backbones, combines them with a **Weighted Soft-Voting Ensemble**, incorporates **Test-Time Augmentation (TTA)**, and provides clinical interpretability using **Grad-CAM++**.

---

## 🔬 Pipeline Architecture

```text
Input MRI (224x224 / 299x299)
        │
        ├── EfficientNetB3 ──────┐
        │   (Two-Phase Tuning)   │
        ├── DenseNet201 ─────────┼──► Soft-Voting Ensemble ──► Final Prediction
        │   (Two-Phase Tuning)   │        (TTA Enabled)
        └── Xception ────────────┘
            (Two-Phase Tuning)
        │
        └──► Grad-CAM++ Visual Explanations (Heatmap Overlays)
```

### Key Technical Highlights
- **Transfer Learning Backbones:** EfficientNetB3, DenseNet201, and Xception.
- **Two-Phase Fine-Tuning:**
  - *Phase 1:* Feature extraction with frozen backbones (`Adam`, `lr=1e-3`).
  - *Phase 2:* Fine-tuning top backbone layers with learning rate scheduling (`CosineAnnealing` / `ReduceLROnPlateau`).
- **Data Augmentation:** Rotation, shift, shear, zoom, and horizontal flip tailored for MRI patterns.
- **Weighted Soft-Voting Ensemble:** Combines model prediction probabilities weighted by validation performance.
- **Test-Time Augmentation (TTA):** Boosts inference robustness by averaging predictions over augmented variants of test samples.
- **Explainable AI (Grad-CAM++):** Visualizes the specific anatomical regions in MRI scans that drive model predictions, aiding clinical decision validation.

---

## 📊 Dataset

The model is trained and evaluated using the **Brain Tumor MRI Dataset** (Kaggle).
- **Classes:** 4 (`glioma`, `meningioma`, `notumor`, `pituitary`)
- **Structure:**
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

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/mirackeser/brain-tumor-mri-classification.git
cd brain-tumor-mri-classification
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Notebook
Launch Jupyter Lab or Jupyter Notebook:
```bash
jupyter notebook brain_tumor_mri_ensemble.ipynb
```

---

## 👥 Authors
- **Muhammet Miraç KESER**
- **Cansu Naz KALKAN**

---

## 📄 License
This project is open-source and licensed under the [MIT License](LICENSE).
