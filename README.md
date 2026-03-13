# AFM-Cell-Aging-Prediction

This repository contains the official implementation for predicting cell aging stages using Atomic Force Microscopy (AFM) data and deep learning models (VAE & MLP Regressor).

## 🌟 Research Highlights
- **Data-driven Aging Analysis**: Leveraging AFM morphology and stiffness data.
- **Generative Modeling**: Utilizing VAE (Variational Autoencoder) for latent space representation of cell aging.
- **Automated Pipeline**: From raw data preprocessing to evaluation and visualization.

## 📂 Repository Structure
```text
├── data/                  # Processed AFM datasets (agg.csv, etc.)
├── upgrade.py             # Core training script for VAE & Regressor
├── evaluation.py          # Model performance evaluation and plotting
├── Inference.py           # Inference script for aging stage prediction
├── PCA.py                 # PCA Dimensionality reduction and visualization
├── plot_training_results.py # Training results visualization
├── data_preprocessing.py  # Raw data cleaning and normalization
└── scaler.pkl             # Pre-trained data scaler