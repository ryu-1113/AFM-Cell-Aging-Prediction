import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import seaborn as sns
import copy
import joblib
import json
from typing import Tuple, List, Dict
import sys
import os

# --- 配置 ---
data_folder = "./data/"
RAW_DATA_PATH = f'agg.csv'
META_FILE_PATH = 'inference_meta.json'
VAE_MODEL_PATH = 'best_vae_model.pth'
REGRESSOR_MODEL_PATH = 'best_regressor_model.pth'
SCALER_PATH = 'scaler.pkl'
RANDOM_SEED = 42

# 检查GPU可用性
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 设置随机种子
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
sns.set_style("whitegrid")
# PDF 格式通常不需要设置 dpi，因为它是矢量图
# plt.rcParams['figure.dpi'] = 150 

# --- 模型结构定义 (必须与 upgrade.py 一致) ---
class VAE(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, latent_dim: int = 2, dropout_rate: float = 0.1):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, latent_dim * 2)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, input_dim)
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        h = self.encoder(x)
        mu, logvar = h.chunk(2, dim=-1)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decoder(z)
        return recon_x, mu, logvar, z

class LatentRegressor(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 32, dropout_rate: float = 0.1):
        super(LatentRegressor, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.network(x)

# --- 辅助函数：加载模型和数据 ---
def load_models_and_data() -> Tuple[pd.DataFrame, VAE, LatentRegressor, StandardScaler, Dict]:
    """加载保存的模型、Scaler、元数据和原始数据"""
    print("INFO: Loading metadata, models, and data...")

    # 1. Load Metadata
    try:
        with open(META_FILE_PATH, 'r') as f:
            meta = json.load(f)
    except FileNotFoundError:
        print(f"ERROR: Metadata file '{META_FILE_PATH}' not found. Exiting.")
        sys.exit(1)

    input_dim = meta.get('input_dim')
    latent_dim = meta.get('latent_dim')
    flip_factor = meta.get('flip_factor', 1)
    feature_cols = meta.get('feature_cols')

    # 2. Load Scaler
    scaler = None
    try:
        scaler = joblib.load(SCALER_PATH)
        print(f"INFO: Scaler '{SCALER_PATH}' loaded successfully.")
    except FileNotFoundError:
        print(f"ERROR: Scaler file '{SCALER_PATH}' not found. Exiting.")
        sys.exit(1)

    # 3. Load Data
    try:
        df_raw = pd.read_csv(os.path.join(data_folder,RAW_DATA_PATH))
    except FileNotFoundError:
        print(f"ERROR: Data file '{RAW_DATA_PATH}' not found. Exiting.")
        sys.exit(1)

    # 4. Load Models
    vae = VAE(input_dim, latent_dim=latent_dim).to(device)
    regressor = LatentRegressor(input_dim=latent_dim).to(device)

    try:
        vae.load_state_dict(torch.load(VAE_MODEL_PATH, map_location=device))
        regressor.load_state_dict(torch.load(REGRESSOR_MODEL_PATH, map_location=device))
        print("INFO: VAE and Regressor models loaded successfully.")
    except FileNotFoundError:
        print(f"ERROR: Model file not found. Check paths: {VAE_MODEL_PATH} and {REGRESSOR_MODEL_PATH}. Exiting.")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: Failed to load model weights. Detail: {e}. Exiting.")
        sys.exit(1)

    vae.eval()
    regressor.eval()

    return df_raw, vae, regressor, scaler, meta

# --- 绘图函数 (修改为保存 PDF) ---

def plot_correlation_heatmap_pdf(df_latent: pd.DataFrame, df_raw: pd.DataFrame, feature_cols: list, z0_col: str = 'z0_aligned'):
    """(1) VAE特征向量与原始特征(中位数/平均值)的关系热力图"""
    print("Generating Correlation Heatmap (PDF)...")
    # 筛选出包含 'mean' 或 'median' 的特征
    target_features = [f for f in feature_cols if 'mean' in f or 'median' in f]
    
    if not target_features:
        print("WARNING: No features with 'mean' or 'median' found for heatmap. Skipping.")
        return

    # 合并数据
    df_corr_input = pd.concat([df_latent[[z0_col, 'z1']], df_raw[target_features]], axis=1)
    
    # 计算 Spearman 相关性
    corr_matrix = df_corr_input.corr(method='spearman')
    # 只关注潜在维度与特征的相关性
    z_corr = corr_matrix.loc[target_features, [z0_col, 'z1']]

    plt.figure(figsize=(10, 12))
    sns.heatmap(z_corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0, fmt='.2f')
    plt.title('Correlation: Latent Dims vs Features (Mean/Median)')
    plt.tight_layout()
    output_file = 'training_plot_1_correlation_heatmap.pdf'
    plt.savefig(output_file, format='pdf', bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()

def plot_latent_space_cycle_pdf(df_latent: pd.DataFrame, z0_col: str = 'z0_aligned'):
    """(2) VAE的潜在空间可视化（按细胞周期着色）"""
    print("Generating Latent Space by Cycle (PDF)...")
    plt.figure(figsize=(10, 7))
    # 定义周期顺序
    cycle_order = ['P4', 'P6', 'P8', 'P10']
    sns.scatterplot(data=df_latent, x=z0_col, y='z1', hue='cycle', 
                    hue_order=cycle_order, palette='viridis', s=50, alpha=0.8)
    plt.title('VAE Latent Space (Calibrated) - Colored by Cycle', fontsize=16)
    plt.xlabel(f'Latent Dimension 1 ({z0_col}) [Low->High Aging]', fontsize=12)
    plt.ylabel('Latent Dimension 2 (z1)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(title='Cell Cycle')
    output_file = 'training_plot_2_latent_space_cycle.pdf'
    plt.savefig(output_file, format='pdf', bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()

def plot_latent_space_probability_pdf(df_latent: pd.DataFrame, z0_col: str = 'z0_aligned'):
    """(3) VAE的潜在空间（按预测着色）"""
    print("Generating Latent Space by Probability (PDF)...")
    plt.figure(figsize=(10, 7))
    sc = plt.scatter(x=df_latent[z0_col], y=df_latent['z1'], c=df_latent['prob_sol_4B'], 
                     cmap='coolwarm', s=50, alpha=0.8)
    cbar = plt.colorbar(sc)
    cbar.set_label("Predicted Senescence Probability", rotation=270, labelpad=15, fontsize=12)
    plt.title('VAE Latent Space (Calibrated) - Colored by Probability', fontsize=16)
    plt.xlabel(f'Latent Dimension 1 ({z0_col}) [Low->High Aging]', fontsize=12)
    plt.ylabel('Latent Dimension 2 (z1)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    output_file = 'training_plot_3_latent_space_probability.pdf'
    plt.savefig(output_file, format='pdf', bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()

def plot_prediction_boxplot_pdf(df_results: pd.DataFrame):
    """(4) 预测结果的箱线图"""
    print("Generating Prediction Boxplot (PDF)...")
    plt.figure(figsize=(10, 5))
    # 定义周期顺序
    cycle_order = ['P4', 'P6', 'P8', 'P10']
    sns.boxplot(data=df_results, x='cycle', y='prob_sol_4B', order=cycle_order, palette="Set2")
    plt.title('VAE + Regressor (Optimized) - Probability Distribution (Boxplot)', fontsize=14)
    plt.ylim(-0.05, 1.05)
    plt.ylabel('Predicted Senescence Probability')
    output_file = 'training_plot_4_prediction_boxplot.pdf'
    plt.savefig(output_file, format='pdf')
    print(f"Saved: {output_file}")
    plt.close()

def plot_prediction_kde_pdf(df_results: pd.DataFrame):
    """(5) 预测结果的密度图分析"""
    print("Generating Prediction KDE Plot (PDF)...")
    plt.figure(figsize=(10, 5))
    # 定义周期顺序
    cycle_order = ['P4', 'P6', 'P8', 'P10']
    for cycle in cycle_order:
        sns.kdeplot(df_results[df_results['cycle'] == cycle]['prob_sol_4B'], 
                    label=cycle, fill=True, alpha=0.4, linewidth=1.5)
    plt.title('VAE + Regressor (Optimized) - Probability Density (KDE)', fontsize=14)
    plt.xlim(-0.05, 1.05)
    plt.xlabel('Predicted Senescence Probability')
    plt.ylabel('Density')
    plt.legend(title='Cell Cycle')
    output_file = 'training_plot_5_prediction_kde.pdf'
    plt.savefig(output_file, format='pdf')
    print(f"Saved: {output_file}")
    plt.close()


# --- 主程序入口 ---
if __name__ == '__main__':
    # 1. 加载所有资源
    df_raw, vae, regressor, scaler, meta = load_models_and_data()
    feature_cols = meta['feature_cols']
    flip_factor = meta['flip_factor']

    # 2. 数据预处理与推理
    print("\n--- Performing Inference on Training Data ---")
    # 标准化
    X_scaled = scaler.transform(df_raw[feature_cols])
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(device)

    with torch.no_grad():
        # a. VAE 编码提取潜在向量
        _, mu_all, _, _ = vae(X_tensor)
        # b. 回归器预测
        probs = regressor(mu_all).cpu().numpy().flatten()
    
    z_mu_all_np = mu_all.cpu().numpy()

    # 3. 构建结果 DataFrame
    df_latent = pd.DataFrame(z_mu_all_np, columns=['z0', 'z1'])
    # 应用轴校准
    df_latent['z0_aligned'] = df_latent['z0'] * flip_factor
    df_latent['prob_sol_4B'] = probs
    df_latent['cycle'] = df_raw['cycle']
    
    # 为了方便绘图，将预测结果也合并回 raw dataframe
    df_results = df_raw.copy()
    df_results['prob_sol_4B'] = probs
    df_results['z0_aligned'] = df_latent['z0_aligned']
    df_results['z1'] = df_latent['z1']

    print("Inference complete. Generating plots...")

    # 4. 生成并保存 PDF 图表
    # (1) 热力图
    plot_correlation_heatmap_pdf(df_latent, df_raw, feature_cols)
    # (2) 潜在空间 (按周期)
    plot_latent_space_cycle_pdf(df_latent)
    # (3) 潜在空间 (按预测)
    plot_latent_space_probability_pdf(df_latent)
    # (4) 箱线图
    plot_prediction_boxplot_pdf(df_results)
    # (5) 密度图
    plot_prediction_kde_pdf(df_results)

    print("\nAll training results plotted and saved as PDF.")