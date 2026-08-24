# SLC26A4 Hearing Loss Prediction Pipeline

This repository contains a machine learning pipeline for predicting Acute Hearing Loss (AHL) events and hearing threshold trends in patients with *SLC26A4* mutations. The workflow integrates representation learning (Autoencoder + K-Means clustering) with regression (ElasticNetCV) and classification (CatBoost) models using a unified 5-fold cross-validation framework.

---

## 1. Prerequisites and Installation

To run the scripts in this project, you need Python 3.8+ installed. You can install all required libraries using the following commands:

```bash
pip install numpy pandas matplotlib scipy scikit-learn torch catboost shap statsmodels imbalanced-learn
```

### Key Libraries Used:
* **PyTorch (`torch`)**: Used for training Autoencoder representations and sequential neural network models (LSTM/Transformers).
* **CatBoost (`catboost`)**: The main classification model predicting AHL peak events.
* **SHAP (`shap`)**: Used for explaining model feature importances.
* **Statsmodels (`statsmodels`) & SciPy (`scipy`)**: Required for calculating and plotting the LOESS (Locally Estimated Scatterplot Smoothing) calibration curve.
* **Scikit-learn (`sklearn`)**: Used for preprocessing, baseline clustering (K-Means), regression (ElasticNetCV), and evaluation metrics.

---

## 2. Directory Structure and Files

```
SLC26A4/
├── S2 Table.csv                # Primary clinical dataset
├── kmeans.py                   # Main modeling & 5-fold cross-validation pipeline
├── Kmeans_elbow.py             # K-Means optimal cluster (K) elbow analysis
├── SHAP_feature.py             # SHAP feature importance analysis and model calibration
├── kmeans_ablation_study.py    # Feature ablation study (32 feature combinations)
├── dataset.py                  # PyTorch custom dataset definitions
├── models.py                   # Neural network architectures (LSTM, Attention, Transformer, MLP)
├── utils.py                    # Preprocessing, data loading (load_s2_table), and split utilities
│
├── figure/                     # Directory for output plots & visualization files
│   ├── tsne/                   # Latent cluster plots and sample waveform plots
│   └── (SHAP & calibration plots)
│
├── log/                        # Directory for log files
│   └── kmeans.log              # Combined execution and evaluation summary log
│
└── model_weight/               # Directory for saved model files & checkpoints
    └── saved_models/           # Saved models across 5 folds (fold_0..fold_4)
```

---

## 3. Detailed File Information

Below is a detailed breakdown of each Python script in the repository, including execution commands and generated outputs.

### A. Primary Modeling Pipeline
#### `kmeans.py`
* **Description**: Main training and evaluation script. Loads clinical data via `load_s2_table('S2 Table.csv')` and applies a unified 5-fold cross-validation split (72% Train / 8% Val / 20% Test). Autoencoder and K-Means models are trained once per fold to annotate 5 cluster trend features. Runs a 50-iteration Kneedle elbow analysis on Fold 0 to validate optimal cluster number $K=5$. Executes three streamlined experiment sections:
  1. **Experiment 1 (Proposed Model & Naive Baselines)**: Evaluates the proposed `elasticnet_catboost` hybrid pipeline alongside `all_positive` (Peak All-True) and `all_negative` (Peak All-False) naive baselines.
  2. **Experiment 2 (Deep Learning Regressors + CatBoost)**: Evaluates deep learning sequence representations (`Transformer`, `MLP`, `Attention`, `LSTM`) paired with CatBoost classifiers.
  3. **Experiment 3 (Standalone ElasticNet Hearing Threshold Regression)**: Evaluates standalone ElasticNetCV regression accuracy (MAE) on hearing threshold decibels.
  Prints per-fold evaluation progress and records a consolidated metric summary (`mean ± std`).
* **Execution Command**:
  ```bash
  python kmeans.py
  ```
* **Output Files / Folders**:
  * `log/kmeans.log` — Consolidated stdout log containing execution details, Fold 0 elbow metrics, per-fold evaluation progress, and the 5-Fold average metric summary table (`mean ± std`).
  * `model_weight/saved_models/fold_{0..4}/` — Serialized model checkpoints for each fold:
    * `autoencoder.pth` — Trained PyTorch representation weights for the fold.
    * `kmeans.pkl` — Serialized K-Means clustering model for the fold.
    * `scaler_mean.npy` / `scaler_std.npy` — Feature scaling arrays for the fold.
    * `elasticnet_freq_{0..5}.pkl` — Frequency-specific regression models.
    * `catboost_catboost.cbm` — Serialized CatBoost classifier for the proposed model.

---

### B. Representation Analysis & Cluster Visualization
#### `Kmeans_elbow.py`
* **Description**: Analyzes the optimal number of clusters ($K$) for grouping patient hearing trends. Trains an Autoencoder, maps representation trajectories, and runs K-Means evaluations for $K=2$ to $10$ to output the elbow plot and cluster visual representations. Loads clinical data using `load_s2_table`.
* **Execution Command**:
  ```bash
  python Kmeans_elbow.py
  ```
* **Output Files / Folders**:
  * `figure/elbow_method.tiff` / `figure/elbow_method.eps` — Plot visualizing Sum of Squared Errors (SSE) over different $K$ counts.
  * `figure/tsne/` — Visualization plots:
    * `pca_latent_clusters.tiff` / `pca_latent_clusters.eps` — PCA plot of patient latent trajectories grouped by K-Means cluster labels.
    * `cluster_{cls}_sample_{sample_idx}.tiff` / `cluster_{cls}_sample_{sample_idx}.eps` — Waveform line plots of raw hearing threshold trajectories of randomly sampled members of each cluster.

---

### C. Explainable AI & Calibration
#### `SHAP_feature.py`
* **Description**: Uses SHAP values to explain feature contributions in the CatBoost classifier. Aggregates values into feature groups (Genotype, Gender, EVA size, etc.) for both global and grouped visualization. Also computes and plots a LOESS Calibration Curve with a 95% Confidence Interval. Loads clinical data using `load_s2_table`.
* **Execution Command**:
  ```bash
  python SHAP_feature.py
  ```
* **Output Files / Folders**:
  * `figure/shap_summary.tiff` / `figure/shap_summary.eps` — Summary beeswarm plot of feature importances.
  * `figure/shap_bar.tiff` / `figure/shap_bar.eps` — Global feature importance bar plot.
  * `figure/shap_summary_grouped.tiff` / `figure/shap_summary_grouped.eps` — Summary beeswarm plot of grouped features.
  * `figure/shap_bar_grouped.tiff` / `figure/shap_bar_grouped.eps` — Grouped feature importance bar plot.
  * `figure/calibration_plot_loess.tiff` / `figure/calibration_plot_loess.eps` — Calibration curve using LOESS smoothing.

---

### D. Ablation Study
#### `kmeans_ablation_study.py`
* **Description**: Systematically evaluates combinations of 5 secondary clinical features (Genotype, Gender, EVA, VA size, Mondini dysplasia) across 32 configuration scenarios to determine their impact on model performance. Loads clinical data using `load_s2_table`.
* **Execution Command**:
  ```bash
  python kmeans_ablation_study.py
  ```
* **Output Files / Folders**:
  * `log/kmeans_ablation_study.log` — Tabulated log summarizing cross-validated metrics (MAE, AUC, F1, specificity, recall, balanced accuracy) for all 32 configuration runs.

---

### E. Support Modules
* **`utils.py`**: Central utility library providing data loading (`load_s2_table`), data cleaning and formatting (`process_acoustic_data`), hearing volatility calculations (`_std_in_time`), trend categorization, peak event logic, PyTorch epoch training/evaluation helpers (`train_epoch`, `evaluate`, `predict`), and unified cross-validation dataset splitting (`k_folder_split_data`).
* **`dataset.py`**: Custom PyTorch dataset definitions (`HearDataset_MAE`, `HearDataset_Peak`) for structuring raw time-series DataFrames into windowed sequence representations.
* **`models.py`**: PyTorch neural network model architectures for sequence modeling (`Transformer_peaks_Model`, `MLP_peaks_Model`, `AModel`, `LSTM_peaks_Model`).

---

## 4. Input Data Format

The pipeline reads clinical data from `S2 Table.csv`. The CSV schema is as follows:

### Header Row
```csv
No.,Gender,Genotype (SLCOne non-LoF6A4),EVA_R,EVA_L,VA size_R,VA size_L,Mondini_R,Mondini_L,Age at exam,R_AC125,R_AC0.25k,R_AC0.5k,R_AC1k,R_AC2k,R_AC4k,R_AC6k,R_AC8k,R_BC0.5k,R_BC1k,R_BC2k,R_BC4k,L_AC125,L_AC0.25k,L_AC0.5k,L_AC1k,L_AC2k,L_AC4k,L_AC6k,L_AC8k,L_BC0.5k,L_BC1k,L_BC2k,L_BC4k
```

### Column Explanations
* **`No.`**: Unique patient identifier (e.g., `S001`). Used to group data points belonging to the same subject.
* **`Gender`**: Patient gender (`F` or `M`). Automatically mapped by `load_s2_table` to `0` (Female) and `1` (Male).
* **`Genotype (SLCOne non-LoF6A4)`**: Genetic mutation class (`Single`, `Two LoF`, `One non-LoF`). Mapped by `load_s2_table` to `2` (`Single`), `3` (`Two LoF`), and `1` (`One non-LoF`).
* **`EVA_R` / `EVA_L`**: Enlarged Vestibular Aqueduct (EVA) status for the right and left ears (`Y` / `N`). Mapped to `1.0` (Yes) and `0.0` (No), defaulting to `-1.0` if empty.
* **`VA size_R` / `VA size_L`**: Measured Vestibular Aqueduct diameter size in mm for the right and left ears. Parsed as numeric (defaults to `-1.0` if empty or missing).
* **`Mondini_R` / `Mondini_L`**: Presence of Mondini dysplasia for the right and left ears (`Y` / `N`). Mapped to `1.0` (Yes) and `0.0` (No), defaulting to `-1.0` if empty.
* **`Age at exam`**: Numeric age (years) of the patient at the time of the clinical check.
* **`R_AC0.25k`, `R_AC0.5k`, `R_AC1k`, `R_AC2k`, `R_AC4k`, `R_AC8k`**: Air conduction thresholds (dB HL) for the right ear at frequencies 250 Hz, 500 Hz, 1 kHz, 2 kHz, 4 kHz, and 8 kHz respectively.
* **`L_AC0.25k`, `L_AC0.5k`, `L_AC1k`, `L_AC2k`, `L_AC4k`, `L_AC8k`**: Air conduction thresholds (dB HL) for the left ear at frequencies 250 Hz, 500 Hz, 1 kHz, 2 kHz, 4 kHz, and 8 kHz respectively.
* **`R_AC125`, `L_AC125`, `R_AC6k`, `L_AC6k`**: Air conduction thresholds (dB HL) at 125 Hz and 6 kHz. *(Unused/ignored in main feature calculations)*.
* **`R_BC0.5k` to `R_BC4k` & `L_BC0.5k` to `L_BC4k`**: Bone conduction thresholds (dB HL) at 500 Hz, 1 kHz, 2 kHz, and 4 kHz. *(Unused/ignored in main feature calculations)*.
