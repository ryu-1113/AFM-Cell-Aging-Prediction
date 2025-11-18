import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple
import os
data_folder = 'data'
# 检查GPU可用性
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# --- 1. 数据加载与准备 ---
def load_and_prepare_data() -> Tuple[pd.DataFrame, torch.Tensor, torch.Tensor, list]:
    csv_filename = f"agg.csv"
    filepath = os.path.join(data_folder, csv_filename)
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        raise FileNotFoundError(f"Error: File '{filepath}' not found. Please ensure it is in the correct directory.")
        
    print(f"Loaded aggregated data with {len(df)} cells from '{filepath}'.")
    
    ignore_cols = ['cell_id', 'cycle']
    feature_cols = [col for col in df.columns if col not in ignore_cols]
    
    if len(feature_cols) == 0:
        raise ValueError("No feature columns found. Check your CSV header.")
    print(f"Found {len(feature_cols)} features for training.") # 应为 18

    # 1. 特征标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[feature_cols])
    
    # 2. 准备软标签 (最符合逻辑的监督信号)
    soft_label_map = {'P4': 0.1, 'P6': 0.3, 'P8': 0.6, 'P10': 0.9}
    df['soft_label'] = df['cycle'].map(soft_label_map)

    # 3. 准备 PyTorch Tensors
    X_all = torch.tensor(X_scaled, dtype=torch.float32)
    y_soft = torch.tensor(df['soft_label'].values, dtype=torch.float32)

    return df, X_all, y_soft, feature_cols

# --- 2. VAE 模型定义 (同上) ---
class VAE(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, latent_dim: int = 2):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim * 2)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar); eps = torch.randn_like(std); return mu + eps * std
    def forward(self, x):
        h = self.encoder(x); mu, logvar = h.chunk(2, dim=-1); z = self.reparameterize(mu, logvar); return self.decoder(z), mu, logvar, z

def vae_loss_function(recon_x, x, mu, logvar):
    recon_loss = nn.functional.mse_loss(recon_x, x, reduction='sum')
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kld

# --- 3. 潜在空间回归器 (Latent Regressor) ---
class LatentRegressor(nn.Module):
    """
    一个简单的MLP，在低维潜在空间上运行，
    其输出目标是回归到软标签 。
    """
    def __init__(self, input_dim: int, hidden_dim: int = 16):
        super(LatentRegressor, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid() # 使用 Sigmoid 将输出压缩到 (0, 1) 范围
        )
    def forward(self, x):
        return self.network(x)

# --- 4. 训练与执行函数 ---

def train_vae(input_dim: int, latent_dim: int, X_all: torch.Tensor) -> VAE:
    """阶段一：训练 VAE"""
    print("Phase 1: Training VAE...")
    model_vae = VAE(input_dim, latent_dim=latent_dim).to(device)
    optimizer_vae = optim.Adam(model_vae.parameters(), lr=1e-3)
    X_all_gpu = X_all.to(device)
    
    for epoch in range(1000): # VAE 训练轮次
        model_vae.train()
        optimizer_vae.zero_grad()
        recon_x, mu, logvar, _ = model_vae(X_all_gpu)
        loss = vae_loss_function(recon_x, X_all_gpu, mu, logvar)
        loss.backward()
        optimizer_vae.step()
        
        if (epoch + 1) % 50 == 0:
            print(f'VAE Epoch [{epoch+1}/1000], Loss: {loss.item() / len(X_all_gpu):.4f}')
    
    model_vae.eval()
    return model_vae

def train_latent_regressor(latent_dim: int, Z_all: torch.Tensor, y_soft: torch.Tensor) -> LatentRegressor:
    """阶段二：在潜在空间上训练软标签回归器"""
    print("\nPhase 2: Training Latent Regressor with Soft Labels...")
    model_regressor = LatentRegressor(input_dim=latent_dim).to(device)
    
    # 我们使用 BCELoss，因为它在处理 (0,1) 之间的概率目标时表现优于 MSE
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model_regressor.parameters(), lr=0.005)
    
    Z_all_gpu = Z_all.to(device)
    y_soft_gpu = y_soft.to(device).view(-1, 1) # 确保形状为 [N, 1]

    for epoch in range(200): # 回归器训练轮次
        model_regressor.train()
        
        # --- 修正点 ---
        # 错误：optimizer_zero_grad()
        # 正确：optimizer.zero_grad()
        optimizer.zero_grad()
        
        # 预测概率
        probs = model_regressor(Z_all_gpu)
        
        # 计算损失 (预测值 vs 软标签)
        loss = criterion(probs, y_soft_gpu)
        
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 50 == 0:
            print(f'Regressor Epoch [{epoch+1}/200], Loss: {loss.item():.4f}')
            
    model_regressor.eval()
    return model_regressor

# --- 5. 可视化函数 ---

def plot_latent_space(df_latent: pd.DataFrame, hue_col: str, title_suffix: str, palette: str = 'viridis'):
    """
    (新功能) 绘制 2D 潜在空间。
    hue_col: 用于着色的列 ('cycle' 或 'prob_sol_4B')
    """
    print(f"Generating latent space plot (colored by {hue_col})...")
    plt.figure(figsize=(10, 7))
    sns.scatterplot(
        data=df_latent,
        x='z0',
        y='z1',
        hue=hue_col,
        palette=palette,
        s=50,
        alpha=0.8,
        legend='full'
    )
    plt.title(f'VAE Latent Space (2D) - {title_suffix}', fontsize=16)
    plt.xlabel('Latent Dimension 1 (z0)', fontsize=12)
    plt.ylabel('Latent Dimension 2 (z1)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(f'solution_4B_latent_space_by_{hue_col}.png', dpi=300)
    plt.close()

def plot_final_distributions(df_results: pd.DataFrame):
    """
    绘制最终的概率分布图 (Box + KDE)
    """
    print("\nGenerating final probability distribution plots...")
    solution_name = 'Solution : VAE + Soft Label Regressor'
    col = 'prob_sol_4B'

    sns.set_style("whitegrid")
    
    # 1. 箱线图 (Box Plot)
    plt.figure(figsize=(10, 5))
    sns.boxplot(data=df_results, x='cycle', y=col, order=['P4', 'P6', 'P8', 'P10'], palette="Set2")
    plt.title(f'{solution_name} - Probability Distribution of Aging (Box Plot)', fontsize=14)
    plt.xlabel('Cycle', fontsize=12); plt.ylabel('预测衰老概率', fontsize=12)
    plt.ylim(-0.05, 1.05); plt.savefig('solution_4B_boxplot.png', dpi=300); plt.close()

    # 2. 核密度估计图 (KDE Plot)
    plt.figure(figsize=(10, 5))
    for cycle in ['P4', 'P6', 'P8', 'P10']:
        sns.kdeplot(df_results[df_results['cycle'] == cycle][col], label=cycle, fill=True, alpha=0.4, linewidth=1.5)
    plt.title(f'{solution_name} - Probability Dense of Aging (KDE Plot)', fontsize=14)
    plt.xlabel('pred_aging_score', fontsize=12); plt.ylabel('dense', fontsize=12)
    plt.xlim(-0.05, 1.05); plt.legend(title='Cycle'); plt.savefig('solution_4B_kdeplot.png', dpi=300); plt.close()
        
    print(f"-> Saved 'solution_4B_boxplot.png' and 'solution_4B_kdeplot.png'")

# --- 6. 主执行函数 ---
if __name__ == '__main__':
    # --- 配置 ---
    INPUT_DIM = 18 # 18 个聚合特征
    LATENT_DIM = 2 # 设为 2 维以便于科学可视化
    
    # --- 1. 加载数据 ---
    df, X_all, y_soft, feature_cols = load_and_prepare_data()
    
    # --- 2. 阶段一：训练 VAE ---
    model_vae = train_vae(INPUT_DIM, LATENT_DIM, X_all)
    
    # --- 3. 提取潜在空间并可视化 (1/2) ---
    with torch.no_grad():
        # 我们使用 mu (均值) 作为细胞在潜在空间的 "稳定" 表示
        _, mu_all, _, _ = model_vae(X_all.to(device))
        
    z_mu_all_np = mu_all.cpu().numpy() # Shape: (400, 2)
    
    # 创建用于绘图的 DataFrame
    df_latent = pd.DataFrame(z_mu_all_np, columns=['z0', 'z1'])
    df_latent['cycle'] = df['cycle']
    
    # 按周期 (Cycle) 着色
    plot_latent_space(df_latent, 'cycle', 'Colored by Cycle', palette='viridis')
    
    # --- 4. 阶段二：训练回归器 ---
    Z_all_tensor = torch.tensor(z_mu_all_np, dtype=torch.float32)
    model_regressor = train_latent_regressor(LATENT_DIM, Z_all_tensor, y_soft)
    
    # --- 5. 获取最终预测并可视化 (2/2) ---
    with torch.no_grad():
        final_probs = model_regressor(Z_all_tensor.to(device)).cpu().numpy().flatten()
    
    df_final_results = df.copy()
    df_final_results['prob_sol_4B'] = final_probs
    
    # 将预测概率添加回潜在空间 DataFrame
    df_latent['prob_sol_4B'] = final_probs
    
    # 按预测概率 (Probability) 着色
    plot_latent_space(df_latent, 'prob_sol_4B', 'Colored by Predicted Probability', palette='coolwarm')
    
    # --- 6. 绘制最终的概率分布图 ---
    plot_final_distributions(df_final_results)
    
    print("\nSolution 4.B experiment complete.")
    print("Please check all 4 generated PNG files for analysis.")