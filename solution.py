import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, List
import os
data_folder = 'data'
solution_folder = 'Solution_Path'
# 检查GPU可用性
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100 # 提高 notebook 中图像的清晰度

# --- 1. 数据加载与准备 ---
def load_and_prepare_data() -> Tuple[pd.DataFrame, torch.Tensor, torch.Tensor, list, StandardScaler]:
    """
    加载 df_agg.csv，进行标准化，准备 Tensors，并返回 scaler 对象。
    """
    csv_filename = f"agg.csv"
    filepath = os.path.join(data_folder, csv_filename)
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        raise FileNotFoundError(f"Error: File '{filepath}' not found.")
        
    print(f"Loaded aggregated data with {len(df)} cells from '{filepath}'.")
    
    ignore_cols = ['cell_id', 'cycle']
    feature_cols = [col for col in df.columns if col not in ignore_cols]
    
    if len(feature_cols) == 0:
        raise ValueError("No feature columns found. Check your CSV header.")
    print(f"Found {len(feature_cols)} features for training.") # 应为 18

    # 1. 特征标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[feature_cols])
    
    # 2. 准备软标签 (使用非线性加速标签)
    print("Using non-linear soft labels for accelerated trend.")
    soft_label_map = {'P4': 0.1, 'P6': 0.2, 'P8': 0.5, 'P10': 0.9}
    df['soft_label'] = df['cycle'].map(soft_label_map)

    # 3. 准备 PyTorch Tensors
    X_all = torch.tensor(X_scaled, dtype=torch.float32)
    y_soft = torch.tensor(df['soft_label'].values, dtype=torch.float32)

    return df, X_all, y_soft, feature_cols, scaler

# --- 2. VAE 模型定义 ---
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
    def __init__(self, input_dim: int, hidden_dim: int = 16):
        super(LatentRegressor, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1), nn.Sigmoid()
        )
    def forward(self, x):
        return self.network(x)

# --- 4. 训练函数 ---
def train_vae(input_dim: int, latent_dim: int, X_all: torch.Tensor) -> VAE:
    """阶段一：训练 VAE"""
    print("\n--- Phase 1: Training VAE ---")
    model_vae = VAE(input_dim, latent_dim=latent_dim).to(device)
    optimizer_vae = optim.Adam(model_vae.parameters(), lr=1e-4)
    X_all_gpu = X_all.to(device)
    
    for epoch in range(1000):
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
    print("\n--- Phase 2: Training Latent Regressor with Soft Labels ---")
    model_regressor = LatentRegressor(input_dim=latent_dim).to(device)
    criterion = nn.BCELoss() # 使用 BCELoss
    optimizer = optim.Adam(model_regressor.parameters(), lr=0.005)
    
    Z_all_gpu = Z_all.to(device)
    y_soft_gpu = y_soft.to(device).view(-1, 1)

    for epoch in range(500):
        model_regressor.train()
        optimizer.zero_grad() # 修正了 NameError
        probs = model_regressor(Z_all_gpu)
        loss = criterion(probs, y_soft_gpu)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 50 == 0:
            print(f'Regressor Epoch [{epoch+1}/500], Loss: {loss.item():.4f}')
            
    model_regressor.eval()
    return model_regressor

# --- 5. 可视化与分析函数 ---

def plot_latent_space(df_latent: pd.DataFrame, hue_col: str, title_suffix: str, palette: str = 'viridis'):
    """
    绘制 2D 潜在空间。
    - 若 hue_col 为连续值（如概率），则使用 plt.scatter 和 plt.colorbar。
    - 若 hue_col 为分类值（如周期），则使用 sns.scatterplot。
    """
    print(f"Generating latent space plot (colored by {hue_col})...")
    plt.figure(figsize=(10, 7))

    # --- 修正逻辑：判断是否为连续值 ---
    if hue_col == 'prob_sol_4B':
        # 1. 连续值处理：使用 plt.scatter 和 plt.colorbar
        print(f"-> Saved 'solution_4B_latent_space_by_{hue_col}' using continuous color bar.")
        # 确定色板
        cmap = 'coolwarm' if hue_col == 'prob_sol_4B' else palette
        
        sc = plt.scatter(
            x=df_latent['z0'],
            y=df_latent['z1'],
            c=df_latent[hue_col],       # 使用 'c' 参数映射连续颜色
            cmap=cmap,                  # 使用 coolwarm 色板
            s=50,
            alpha=0.8
        )
        
        # 添加连续颜色条
        cbar = plt.colorbar(sc)
        cbar.set_label("Predicted Senescence Probability", rotation=270, labelpad=15, fontsize=12)
        
        # 不使用 plt.legend()，但需要确保轴标签存在
        plt.xlabel('Latent Dimension 1 (z0)', fontsize=12)
        plt.ylabel('Latent Dimension 2 (z1)', fontsize=12)
        
    else:
        # 2. 分类值处理：使用 sns.scatterplot 和标准图例
        print(f"-> Saved 'solution_4B_latent_space_by_{hue_col}' using color bar.")
        import seaborn as sns # 确保 sns 在此作用域内可用
        sns.scatterplot(
            data=df_latent, x='z0', y='z1',
            hue=hue_col, 
            palette=palette, 
            s=50, 
            alpha=0.8, 
            legend='full'
        )
        plt.legend(title=hue_col, loc='upper right')
        plt.xlabel('Latent Dimension 1 (z0)', fontsize=12) # 确保分类图也有标签
        plt.ylabel('Latent Dimension 2 (z1)', fontsize=12)


    plt.title(f'VAE Latent Space (2D) - {title_suffix}', fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # 使用 bbox_inches='tight' 确保颜色条或图例被完整保存
    plt.savefig(os.path.join(solution_folder,f'solution_4B_latent_space_by_{hue_col}.png'), dpi=300, bbox_inches='tight')
    plt.close()


def plot_final_distributions(df_results: pd.DataFrame):
    """绘制最终的概率分布图 (Box + KDE)"""
    print("\nGenerating final probability distribution plots...")
    solution_name = 'solution: VAE + Soft Label Regressor'
    col = 'prob_sol_4B'
    sns.set_style("whitegrid")
    
    plt.figure(figsize=(10, 5))
    sns.boxplot(data=df_results, x='cycle', y=col, order=['P4', 'P6', 'P8', 'P10'], palette="Set2")
    plt.title(f'{solution_name} - Probability Distribution of Aging (Box Plot)', fontsize=14)
    plt.xlabel('Cycle', fontsize=12); plt.ylabel('Pred_aging_score', fontsize=12)
    plt.ylim(-0.05, 1.05); plt.savefig(os.path.join(solution_folder,'solution_boxplot.png'), dpi=300); plt.close()

    plt.figure(figsize=(10, 5))
    for cycle in ['P4', 'P6', 'P8', 'P10']:
        sns.kdeplot(df_results[df_results['cycle'] == cycle][col], label=cycle, fill=True, alpha=0.4, linewidth=1.5)
    plt.title(f'{solution_name} - Probability Dense of Aging (KDE Plot)', fontsize=14)
    plt.xlabel('Pred_aging_score', fontsize=12); plt.ylabel('Dense', fontsize=12)
    plt.xlim(-0.05, 1.05); plt.legend(title='Cycle'); plt.savefig(os.path.join(solution_folder,'solution_kdeplot.png'), dpi=300); plt.close()
        
    print(f"-> Saved 'solution_boxplot.png' and 'solution_kdeplot.png'")

# --- 6. 新增：可解释性分析函数 ---

def analyze_latent_correlations(df: pd.DataFrame, z_data: np.ndarray, feature_cols: list):
    """
    (方法一) 计算潜在变量与原始特征的相关性，并绘制热力图。
    """
    print("\nRunning Analysis Method 1: Latent-Feature Correlation...")
    df_z = pd.DataFrame(z_data, columns=['z0', 'z1'])
    df_features = df[feature_cols]
    df_combined = pd.concat([df_z, df_features], axis=1)
    
    # 仅计算 z0, z1 与 18 个特征列的相关性
    correlation_matrix = df_combined.corr(method='spearman')
    z_corr = correlation_matrix.loc[feature_cols, ['z0', 'z1']]
    
    plt.figure(figsize=(8, 10))
    sns.heatmap(z_corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0, fmt='.2f')
    plt.title('Correlation between Latent Dimensions and Physical Features')
    plt.tight_layout()
    plt.savefig(os.path.join(solution_folder,'analysis_1_correlation_heatmap.png'), dpi=300)
    plt.close()
    print("-> Saved 'analysis_1_correlation_heatmap.png'")

def plot_feature_overlay(df_latent: pd.DataFrame, df_original: pd.DataFrame, features_to_plot: list):
    """
    (方法二) 绘制潜在空间，但根据物理特征的值进行着色。
    [修正] 使用 plt.scatter 和 plt.colorbar 来创建连续的颜色条图例，
    而不是使用 seaborn 的离散图例。
    """
    print("\nRunning Analysis Method 2: Feature Overlay Plots (Corrected)...")
    
    for feature in features_to_plot:
        if feature not in df_original.columns:
            print(f"Warning: Feature '{feature}' not found in df. Skipping overlay plot.")
            continue
            
        # 1. 创建画布
        #    (figsize 调整为更适合带颜色条的图表)
        plt.figure(figsize=(10, 7)) 
        
        # 2. [核心修正] 
        #    不使用 sns.scatterplot(hue=...)
        #    而是使用 plt.scatter(c=...)
        sc = plt.scatter(
            x=df_latent['z0'],             # X 轴数据
            y=df_latent['z1'],             # Y 轴数据
            c=df_original[feature],      # 使用 'c' 参数映射连续颜色
            cmap='viridis',              # 'viridis' 色板 (同之前)
            s=50,                        # 点的大小
            alpha=0.8                    # 透明度
        )
        
        # 3. [核心修正] 添加连续颜色条
        #    plt.colorbar() 会自动创建一个与 'sc' 匹配的图例
        cbar = plt.colorbar(sc)
        cbar.set_label(feature, rotation=270, labelpad=15, fontsize=12) # 为颜色条添加标签
        
        # 4. 添加标题和标签 (matplotlib 需要手动添加)
        plt.title(f'Latent Space colored by {feature}', fontsize=16)
        plt.xlabel('Latent Dimension 1 (z0)', fontsize=12)
        plt.ylabel('Latent Dimension 2 (z1)', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.5)
        
        # 5. 保存图像
        #    bbox_inches='tight' 仍然是好习惯，确保标题和颜色条标签不被裁切
        plt.savefig(
            os.path.join(solution_folder,f'analysis_2_latent_map_{feature}.png'), 
            dpi=300, 
            bbox_inches='tight' 
        )
        plt.close()
        
    print(f"-> Saved overlay plots for {len(features_to_plot)} features.")

def analyze_decoder_traversal(model_vae: VAE, scaler: StandardScaler, feature_cols: List[str], latent_dim: int):
    """
    (方法三) 沿着 z0 轴遍历，观察解码出的物理特征如何变化。
    """
    print("\nRunning Analysis Method 3: Decoder Traversal...")
    model_vae.eval()
    
    # 1. 创建遍历向量: z0 从 -3 到 3, 其他 z 轴固定为 0
    z0_steps = np.linspace(-3, 3, 20)
    z_traversal = np.zeros((20, latent_dim))
    z_traversal[:, 0] = z0_steps
    
    z_tensor = torch.tensor(z_traversal, dtype=torch.float32).to(device)
    
    # 2. 解码
    with torch.no_grad():
        recon_x_scaled = model_vae.decoder(z_tensor).cpu().numpy()
        
    # 3. 逆标准化
    recon_x_original = scaler.inverse_transform(recon_x_scaled)
    
    # 4. 转换为 DataFrame
    df_recon = pd.DataFrame(recon_x_original, columns=feature_cols)
    df_recon['z0_value'] = z0_steps
    
    # 5. 绘制趋势线 (只绘制 6 个 _mean 特征)
    cols_to_plot = [c for c in feature_cols if c.endswith('_mean')]
    if not cols_to_plot: # 如果没有 _mean, 就用前 6 个
        cols_to_plot = feature_cols[:6]

    plt.figure(figsize=(12, 7))
    for col in cols_to_plot:
        y_vals = df_recon[col]
        # 归一化 (0-1) 以便在同一张图上显示趋势
        y_norm = (y_vals - y_vals.min()) / (y_vals.max() - y_vals.min())
        plt.plot(df_recon['z0_value'], y_norm, label=col, linewidth=2, marker='o', markersize=4)
        
    plt.title('Reconstructed Physical Features along the Aging Axis (z0)')
    plt.xlabel('Latent Dimension z0 (<- Young | Old ->)')
    plt.ylabel('Normalized Feature Value (0 to 1 scale)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2)
    plt.grid(True, linestyle='--')
    plt.tight_layout()
    plt.savefig(os.path.join(solution_folder,'analysis_3_decoder_traversal.png'), dpi=300)
    plt.close()
    print("-> Saved 'analysis_3_decoder_traversal.png'")

# --- axis flipping func ---




def align_aging_axis(df_latent: pd.DataFrame, latent_col: str = 'z0', aging_score_col: str = 'prob_sol_4B') -> pd.DataFrame:
    """
    检查潜在维度 (z0) 与衰老分数之间的相关性，并在必要时翻转 z0 轴。

    Args:
        df_latent: 包含潜在维度和预测衰老分数的 DataFrame。
        latent_col: 潜在维度列名，默认为 'z0'。
        aging_score_col: 衰老分数（如概率）的列名。
    
    Returns:
        DataFrame，其中包含一个已校准的新列 'z0_aligned'。
    """
    from scipy.stats import spearmanr
    # 1. 计算 Spearman 秩相关系数
    # 秩相关更适合潜在空间，因为它不假设线性关系，更侧重于排序关系。
    correlation, p_value = spearmanr(df_latent[latent_col], df_latent[aging_score_col])
    
    print(f"orginal {latent_col} and aging-score  spearman: {correlation:.4f}")

    # 2. 检查相关系数，并进行轴校准
    if correlation < 0:
        # 如果相关系数为负，表示 z0 轴与衰老方向相反（年轻 -> 衰老 对应 z0 负向 -> z0 正向）
        # 需要翻转 z0 轴。
        df_latent['z0'] = -df_latent[latent_col]
        print(f"axis flipping: minus spearman {latent_col} was flipped。")
    else:
        # 如果相关系数为正，方向正确（年轻 -> 衰老 对应 z0 负向 -> z0 正向）
        df_latent['z0'] = df_latent[latent_col]
        print(f"no axix flipping, using original {latent_col} .")
        
    return df_latent 

# 示例调用
# df_latent_aligned = align_aging_axis(df_latent, latent_col='z0', aging_score_col='Pred_aging_score')

# 后续所有的分析（例如 plot_latent_space、analysis_3_decoder_traversal）都应该使用 'z0_aligned' 替代 'z0'。

# --- 7. 主执行函数 ---
if __name__ == '__main__':
    # --- 配置 --- 
    LATENT_DIM = 2 # 设为 2 维以便于科学可视化
    
    # --- 1. 加载数据 ---
    df, X_all, y_soft, feature_cols, scaler = load_and_prepare_data()
    INPUT_DIM = len(feature_cols)
    
    # --- 2. 阶段一：训练 VAE ---
    model_vae = train_vae(INPUT_DIM, LATENT_DIM, X_all)
    
    # --- 3. 提取潜在空间 ---
    with torch.no_grad():
        _, mu_all, _, _ = model_vae(X_all.to(device))
    z_mu_all_np = mu_all.cpu().numpy() # Shape: (400, 2)
    
    df_latent = pd.DataFrame(z_mu_all_np, columns=['z0', 'z1'])
    df_latent['cycle'] = df['cycle']
    
    # --- 4. 阶段二：训练回归器 ---
    Z_all_tensor = torch.tensor(z_mu_all_np, dtype=torch.float32)
    model_regressor = train_latent_regressor(LATENT_DIM, Z_all_tensor, y_soft)
    
    # --- 5. 获取最终预测 ---
    with torch.no_grad():
        final_probs = model_regressor(Z_all_tensor.to(device)).cpu().numpy().flatten()
    
    df_final_results = df.copy()
    df_final_results['prob_sol_4B'] = final_probs
    df_latent['prob_sol_4B'] = final_probs
    df_latent = align_aging_axis(df_latent=df_latent)
    # --- 6. 运行标准可视化 ---
    print("\n--- Running Standard Visualizations ---")
    plot_latent_space(df_latent, 'cycle', 'Colored by Cycle', palette='viridis')
    plot_latent_space(df_latent, 'prob_sol_4B', 'Colored by Predicted Probability', palette='coolwarm')
    plot_final_distributions(df_final_results)
    
    # --- 7. 运行可解释性分析 ---
    print("\n--- Running Interpretability Analyses ---")
    
    # (方法一)
    analyze_latent_correlations(df, z_mu_all_np, feature_cols)
    
    # (方法二)
    key_features = ['height_mean', 'roughness_mean', 'adhesion_mean', 'elastic_modulus_mean']
    plot_feature_overlay(df_latent, df, key_features)
    
    # (方法三)
    analyze_decoder_traversal(model_vae, scaler, feature_cols, LATENT_DIM)
    
    print(f"\nExperiment complete. All {5 + len(key_features)} analysis images have been saved.")