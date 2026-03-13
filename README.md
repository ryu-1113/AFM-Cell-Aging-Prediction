# AFM-Cell-Aging-Prediction

This repository provides a deep learning framework to predict and quantify cell senescence levels based on biophysical features extracted via Atomic Force Microscopy (AFM). The project utilizes a **Variational Autoencoder (VAE)** for non-linear dimensionality reduction and a **Multi-Layer Perceptron (MLP)** for aging probability regression.

## 📌 Project Overview
Traditional linear methods often fail to distinguish cellular aging stages due to high data overlap in physical properties. Our model overcomes this by mapping 18-dimensional biophysical statistics into a 2D latent manifold, successfully capturing the continuous trajectory of cell senescence even under external environmental stressors (e.g., Hypoxia).

## 📂 Directory Structure

```text
├── data/
│   ├── agg.csv                # Primary training data (Passages P4, P6, P8, P10)
│   └── new_p6_data.csv        # Independent test set (Normoxia vs. Hypoxia conditions)
├── result/                    # Output directory for generated plots and PDF reports
├── upgrade.py                 # Training script: Trains VAE + Regressor and saves models
├── evaluation.py              # Evaluation: Processes new data and validates model on downstream tasks
├── comparison.py              # Benchmarking: Compares VAE+MLP against PCA linear baseline
├── plot_training_results.py   # Visualization: Latent space plots, heatmaps, and distributions
├── Inference.py               # Quick Inference: Loads models to predict on validation sets
├── PCA.py                     # Baseline: PCA processing and visualization of raw data
├── best_vae_model.pth         # Saved VAE weights
├── best_regressor_model.pth   # Saved Regressor weights
├── inference_meta.json        # Model hyperparameters and metadata
├── scaler.pkl                 # Fitted StandardScaler for feature normalization
└── requirements.txt           # Python dependency list
```

## 🚀 Getting Started

### 1. Environment Setup

Install the necessary dependencies using the provided `requirements.txt` file:
```bash
pip install -r requirements.txt
```
### 2. Quick Reproducibility

To reproduce the exact results and figures presented in the paper, it is highly recommended to use the pre-trained model weights provided in this repository. You can directly run the evaluation or comparison scripts:
Downstream Tasks: Run `python evaluation.py` or `python comparison.py` to verify model performance on independent test dataset (e.g., Hypoxia stress).

Visualization: Run `python plot_training_results.py` to generate the latent space maps and heatmaps using the existing model.

### 3. Training from Scratch

If you wish to retrain the model, run:

```bash
python upgrade.py
```
Note: While a fixed seed (RANDOM_SEED = 42) is used, slight variations in the latent space orientation may occur due to the stochastic nature of VAE initialization and hardware-level non-determinism. For consistent visualization, the provided weights are preferred.

## 🧪 Model Methodology

### Feature Engineering & Preprocessing
The model utilizes 6 key biophysical parameters measured via AFM: **Adhesion, Elastic Modulus, Height, Roughness, Length, and Width**. 

* **Feature Aggregation**: Raw AFM measurements for each cell are aggregated into three statistical metrics: **Mean, Median, and Standard Deviation**.
* **Input Vector**: This transformation results in an **18-dimensional input vector** (6 features × 3 statistics) for each sample.
* **Normalization**: Features are standardized via a fitted `StandardScaler` (saved as `scaler.pkl`) to ensure stable model convergence.

### Two-Stage Deep Learning Architecture

#### Stage 1: Variational Autoencoder (VAE)
The unsupervised VAE is designed to learn a robust, non-linear latent representation of cell biophysics.
* **Architecture**: The Encoder compresses the 18D input through hidden layers (18D Input → Linear(64) → ReLU → Dropout(0.1) → Linear(4)) into a **2D Latent Space** ($z=2$). The Decoder mirrors this structure to reconstruct the original features.

#### Stage 2: Latent Regressor (MLP)
A supervised regression model is built on top of the frozen, calibrated latent manifold.
* **Architecture**: A Multi-Layer Perceptron (2D Input → Linear(32) → ReLU → Dropout(0.1) → Linear(1) → Sigmoid.).
* **Output**: The model outputs a **Senescence Probability** (ranging from 0.0 to 1.0), representing the predicted physiological age and senescence state of the cell.

### Data Partitioning Strategy
* **Internal Split**: During training, the primary dataset (`agg.csv`, containing passages P4, P6, P8, P10) is partitioned using a random 80/20 split. This allows the model to capture the continuous heterogeneity and natural physical transitions within and across cell passages.
* **Independent External Testing**: To rigorously validate "Out-of-Distribution" (OOD) generalization, the model is evaluated on a completely independent downstream dataset (`new_p6_data.csv`). This dataset introduces new environmental stressors (**Hypoxia vs. Normoxia** at P6) not seen during the training phase, proving the model's ability to detect environmentally accelerated aging signatures rather than merely memorizing passage numbers.
