# Alzheimer's Detection in Brain MRI — Sequential vs. Residual CNN

Comparison of two convolutional neural networks **built from scratch** (no pretrained weights) for classifying brain MRI scans into four Alzheimer's severity stages. Both models were trained under identical conditions, so the performance differences can be attributed to the architecture design.

**Key result:** the residual network reached **87% accuracy** (vs. 77%) and raised recall on the hardest, most clinically relevant class, *Very Mild Demented*, from **0.60 to 0.93**. It did this with **27% fewer parameters**.

> Master's thesis (TFM) — Master's in Big Data Analysis & Visual Analytics, Universidad Internacional de La Rioja (UNIR), 2026.

---

## Problem

Early-stage Alzheimer's produces very subtle changes in brain structure that are hard to tell apart from normal aging. Many published models report high accuracy only on binary tasks (healthy vs. advanced disease) or compare architectures under different training setups. This project runs a controlled comparison on a **4-class** problem, with a focus on the early-detection boundary.

## Dataset

[Alzheimer's Multiclass Dataset (Equal and Augmented)](https://www.kaggle.com/datasets/aryansinghal10/alzheimers-multiclass-dataset-equal-and-augmented) by A. Singhal, published on Kaggle under a CC BY 4.0 license.

| Class | Images | Share |
|---|---|---|
| NonDemented | 12,800 | 29.1% |
| VeryMildDemented | 11,200 | 25.5% |
| MildDemented | 10,000 | 22.7% |
| ModerateDemented | 10,000 | 22.7% |
| **Total** | **44,000** | |

The images are axial, skull-stripped MRI slices. The data is **not stored in this repository**; the scripts download it automatically with `kagglehub`.

## Experimental setup (identical for both models)

| Parameter | Value |
|---|---|
| Input size | 224 × 224 × 3 |
| Split | 80% train / 20% validation, stratified, seed 123 |
| Optimizer | Adam, learning rate 1e-3 |
| Loss | Categorical cross-entropy |
| Batch size / epochs | 32 / 15 |
| Weights | Trained from scratch (He-normal initialization) |

## Architectures

**Sequential CNN** (`src/sequential_cnn.py`)
- Four linear Conv2D blocks with 32 → 64 → 128 → 256 filters
- Each block: BatchNorm, ReLU, MaxPooling and Dropout (0.25 → 0.30)
- GlobalAveragePooling → Dense(128) → Dropout(0.5) → Softmax(4)
- **422,788** trainable parameters

**Residual CNN** (`src/residual_cnn.py`)
- 7×7 stride-2 stem convolution, followed by two custom residual blocks with 64 and 128 filters
- Each residual block has two 3×3 convolutions plus a skip connection (1×1 projection when channel counts differ)
- GlobalAveragePooling → Dense(128) → Dropout(0.5) → Softmax(4)
- **310,544** trainable parameters

## Results

Evaluated on the 8,800-image validation set.

| Metric | Sequential | Residual |
|---|---|---|
| **Accuracy** | 0.77 | **0.87** |
| **Weighted F1** | 0.76 | **0.87** |
| F1 — NonDemented | 0.71 | **0.93** |
| F1 — VeryMildDemented | 0.59 | **0.81** |
| F1 — MildDemented | 0.79 | **0.92** |
| F1 — ModerateDemented | **0.99** | 0.82 |
| Trainable parameters | 422,788 | **310,544** |

### Early-detection boundary (NonDemented ↔ VeryMildDemented)

| Error type | Sequential | Residual |
|---|---|---|
| Very mild cases predicted as healthy (false negatives) | 274 | **136** |
| Healthy cases predicted as very mild (false positives) | 785 | **100** |

### Trade-off
The sequential network classified *Moderate Demented* almost perfectly (recall 1.00). The residual network misclassified 481 of those cases as *Very Mild*, so its recall on that class dropped to 0.69.

The recommendation is to use the residual network for screening and early detection. The sequential network remains useful as a second check for advanced cases.

<!-- Add figures, e.g.:
![Confusion matrix – residual](figures/confusion_matrix_residual.png)
![Learning curves](figures/learning_curves.png)
-->

## Limitations
- **Possible data leakage.** The dataset was balanced with augmentation before publication, and it has no patient IDs. Augmented copies of the same scan may therefore appear in both the training and validation sets, which could inflate the reported metrics.
- **No independent test set.** Metrics come from the validation split, with a single fixed seed and no cross-validation.
- **Hyperparameters were fixed for comparability**, not tuned per model.
- **No statistical significance test** was run on the 10-point difference. McNemar's test would be the appropriate one.

## Future work
- Patient-level splits and external validation (e.g., ADNI)
- K-fold cross-validation with confidence intervals
- Grad-CAM heatmaps to show which regions drive each prediction
- 3D CNNs on full MRI volumes

## How to run

```bash
pip install -r requirements.txt
python src/sequential_cnn.py
python src/residual_cnn.py
```

A GPU is recommended. The scripts were developed in Google Colab.

## Tech stack
Python · TensorFlow / Keras · scikit-learn · NumPy · Matplotlib · Seaborn · Google Colab

## Authors
- **Maria Paula Guaca Campo**: sequential network, data pipeline, EDA and comparative analysis. [LinkedIn](https://linkedin.com/in/mariapaulaguacacampo)
- **Mateo Arteaga**: residual network. [Original repository](https://github.com/mateoart1014/master_tfm)

Advisor: Carmen Pérez Gandia, UNIR.
