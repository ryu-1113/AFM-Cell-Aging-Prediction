import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Dict
import os
data_folder = 'data'
# 检查GPU可用性
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# --- 步骤 1: 数据加载、特征工程与标准化 ---
def load_and_prepare_data() -> Tuple[pd.DataFrame, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list]:
    """
    加载聚合后的 400 个细胞数据，进行标准化，并准备 PyTorch Tensor。
    """
    csv_filename = f"agg.csv"
    csv_path = os.path.join(data_folder, csv_filename)
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"Error: Required file '{csv_path}' not found. Please ensure your aggregated data is saved as 'cell_data.csv'.")
        
    print(f"Loaded aggregated data with {len(df)} cells.")
    
    # 确定特征列 (排除 'cell_id' 和 'cycle')
    ignore_cols = ['cell_id', 'cycle']
    feature_cols = [col for col in df.columns if col not in ignore_cols]
    
    if len(feature_cols) != 18:
        print(f"Warning: Expected 18 features (6x mean/std/median), but found {len(feature_cols)}.")
        
    # 1. 特征标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[feature_cols])
    
    # 2. 标签处理
    # 统一将 P4, P6, P8, P10 映射为软标签和硬标签
    soft_label_map = {'P4': 0.1, 'P6': 0.3, 'P8': 0.7, 'P10': 0.9} 
    hard_label_map = {'P4': 0, 'P10': 1} 
    
    df['soft_label'] = df['cycle'].map(soft_label_map)
    
    # 准备 PyTorch Tensors
    X_all = torch.tensor(X_scaled, dtype=torch.float32)
    y_soft = torch.tensor(df['soft_label'].values, dtype=torch.float32)
    
    # 用于方案 1 的数据 (P4 vs P10)
    df_p4_p10 = df[df['cycle'].isin(['P4', 'P10'])].copy()
    X_p4_p10 = torch.tensor(scaler.transform(df_p4_p10[feature_cols]), dtype=torch.float32)
    y_p4_p10 = torch.tensor(df_p4_p10['cycle'].map(hard_label_map).values, dtype=torch.float32)

    return df, X_all, y_soft, X_p4_p10, y_p4_p10, feature_cols

# --- 通用 MLP 模型 (方案 1, 2) ---
class SimpleMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim1=64, hidden_dim2=32):
        super(SimpleMLP, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim1),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.network(x)

# --- 通用训练循环 (方案 1, 2) ---
def train_mlp(model, X_train, y_train, epochs=150, lr=0.001):
    model.to(device)
    # 将 y_train 转换为 (N, 1) 形状
    y_train = y_train.to(device).view(-1, 1)
    
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    print(f"Training on {len(X_train)} samples...")
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_train.to(device))
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 50 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
    
    model.eval()
    return model

# --- 方案 1: 代理标签监督学习 (Proxy-Label MLP) ---
def run_solution_1(df, X_all, X_p4_p10, y_p4_p10, input_dim):
    print("\n--- Running Solution 1: Proxy-Label MLP (P4 vs P10) ---")
    
    model_1 = SimpleMLP(input_dim)
    model_1 = train_mlp(model_1, X_p4_p10, y_p4_p10, epochs=150)
    
    with torch.no_grad():
        X_all = X_all.to(device)
        probs = model_1(X_all).cpu().numpy().flatten()
    
    df_res = df.copy()
    df_res['prob_sol_1'] = probs
    
    print("S1 Mean Probabilities:\n", df_res.groupby('cycle')['prob_sol_1'].mean().reindex(['P4', 'P6', 'P8', 'P10']))
    
    return df_res, model_1

# --- 方案 2: 概率标签监督学习 (Probabilistic-Label MLP) ---
def run_solution_2(df, X_all, y_soft, input_dim):
    print("\n--- Running Solution 2: Probabilistic-Label MLP (All Data) ---")
    
    model_2 = SimpleMLP(input_dim)
    model_2 = train_mlp(model_2, X_all, y_soft, epochs=200, lr=0.0005)
    
    with torch.no_grad():
        X_all = X_all.to(device)
        probs = model_2(X_all).cpu().numpy().flatten()
        
    df_res = df.copy()
    df_res['prob_sol_2'] = probs
    
    print("S2 Mean Probabilities:\n", df_res.groupby('cycle')['prob_sol_2'].mean().reindex(['P4', 'P6', 'P8', 'P10']))
    
    return df_res, model_2

# --- 方案 3: GMM 无监督聚类 ---
def run_solution_3(df, X_all):
    print("\n--- Running Solution 3: GMM (Unsupervised) ---")
    X_all_np = X_all.cpu().numpy() 
    
    gmm = GaussianMixture(n_components=2, random_state=42, n_init=10, max_iter=500)
    gmm.fit(X_all_np)
    
    # 预测概率 (probs_gmm shape: N x 2)
    probs_gmm = gmm.predict_proba(X_all_np)
    
    df_res = df.copy()
    df_res['gmm_prob_0'] = probs_gmm[:, 0]
    df_res['gmm_prob_1'] = probs_gmm[:, 1]
    
    # 关键：确定哪个簇是 "衰老" 簇。期望衰老簇在 P10 中的概率更高。
    # 检查簇 1 的平均概率是否随周期增长
    means = df_res.groupby('cycle')['gmm_prob_1'].mean().reindex(['P4', 'P10'])
    
    if means['P10'] > means['P4']:
        # 簇 1 是衰老簇
        df_res['prob_sol_3'] = df_res['gmm_prob_1']
    else:
        # 簇 0 是衰老簇
        df_res['prob_sol_3'] = df_res['gmm_prob_0']
        
    print("S3 Mean Probabilities:\n", df_res.groupby('cycle')['prob_sol_3'].mean().reindex(['P4', 'P6', 'P8', 'P10']))
        
    return df_res, gmm

# --- 方案 4: VAE + MLP ---

# VAE 模型定义 (略微简化以节省空间，假设在主脚本中)
class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, latent_dim=3): # 使用 3D 潜在空间
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim * 2) 
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim), nn.ReLU(),
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

def vae_loss_function(recon_x, x, mu, logvar):
    recon_loss = nn.functional.mse_loss(recon_x, x, reduction='sum')
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kld

class LatentMLP(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=16):
        super(LatentMLP, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.network(x)

def run_solution_4(df, X_all, X_p4_p10, y_p4_p10, input_dim, latent_dim=4):
    print("\n--- Running Solution 4: VAE + Latent MLP ---")
    
    # 1. 训练 VAE
    model_vae = VAE(input_dim, latent_dim=latent_dim).to(device)
    optimizer_vae = optim.Adam(model_vae.parameters(), lr=1e-3)
    X_all_gpu = X_all.to(device)
    
    model_vae.train()
    for epoch in range(1000):
        optimizer_vae.zero_grad()
        recon_x, mu, logvar, _ = model_vae(X_all_gpu)
        loss = vae_loss_function(recon_x, X_all_gpu, mu, logvar)
        loss.backward()
        optimizer_vae.step()
        if (epoch + 1) % 100 == 0:
             print(f'VAE Epoch [{epoch+1}/1000], Loss: {loss.item() / len(X_all_gpu):.4f}')
            
    # 2. 提取潜在空间表示
    model_vae.eval()
    with torch.no_grad():
        _, mu_all, _, _ = model_vae(X_all.to(device))
    z_mu_all = mu_all.cpu().numpy()
    
    # 3. 训练潜在空间分类器 (使用 P4 vs P10 代理标签)
    p4_p10_indices = df[df['cycle'].isin(['P4', 'P10'])].index
    z_train = torch.tensor(z_mu_all[p4_p10_indices], dtype=torch.float32)
    y_train = y_p4_p10 
    
    print("Training classifier on latent space...")
    model_latent_mlp = LatentMLP(input_dim=latent_dim)
    model_latent_mlp = train_mlp(model_latent_mlp, z_train, y_train, epochs=150)

    # 4. 预测所有细胞
    with torch.no_grad():
        z_all_tensor = torch.tensor(z_mu_all, dtype=torch.float32).to(device)
        probs = model_latent_mlp(z_all_tensor).cpu().numpy().flatten()
    
    df_res = df.copy()
    df_res['prob_sol_4'] = probs
    
    print("S4 Mean Probabilities:\n", df_res.groupby('cycle')['prob_sol_4'].mean().reindex(['P4', 'P6', 'P8', 'P10']))
    
    return df_res, model_vae, model_latent_mlp

# --- 步骤 2: 结果可视化 ---
def plot_results(df_results: pd.DataFrame):
    """
    使用小提琴图 (Violin Plot) 可视化所有方案的概率分布。
    """
    print("\nGenerating final comparison plot...")
    
    prob_cols = [col for col in df_results.columns if col.startswith('prob_sol_')]
    
    df_long = pd.melt(df_results, 
                      id_vars=['cycle'], 
                      value_vars=prob_cols, 
                      var_name='Solution', 
                      value_name='Senescence_Probability')
    
    solution_map = {
        'prob_sol_1': '1. MLP (P4/P10 Hard)',
        'prob_sol_2': '2. MLP (Soft Label)',
        'prob_sol_3': '3. GMM (Unsupervised)',
        'prob_sol_4': '4. VAE + MLP'
    }
    df_long['Solution'] = df_long['Solution'].map(solution_map)

    plt.figure(figsize=(16, 8))
    sns.violinplot(
        data=df_long,
        x='cycle',
        y='Senescence_Probability',
        hue='Solution',
        order=['P4', 'P6', 'P8', 'P10'], 
        inner='quartile', 
        palette='muted'
    )
    plt.title('Comparison of Senescence Probability Distributions by Model', fontsize=16)
    plt.xlabel('Cell Cycle', fontsize=12)
    plt.ylabel('Predicted Senescence Probability', fontsize=12)
    plt.ylim(-0.05, 1.05)
    plt.legend(title='Model Solution', loc='upper left')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    plt.savefig('solution_comparison_violin.png', dpi=300)
    print("Saved 'solution_comparison_violin.png'")

# --- 补充步骤 2: 单独可视化每种方案的效果 ---

def plot_individual_results(df_results: pd.DataFrame):
    """
    为每种方案单独生成可视化图表，展示衰老概率的分布和趋势。
    使用箱线图和核密度估计图。
    """
    print("\nGenerating individual solution visualization plots...")
    
    solution_maps = {
        'prob_sol_1': 'solution 1: MLP (P4/P10 Hard Label)',
        'prob_sol_2': 'solution 2: MLP (Soft Label)',
        'prob_sol_3': 'solution 3: GMM (Unsupervised)',
        'prob_sol_4': 'solution 4: VAE + MLP'
    }
    
    # 获取所有需要可视化的概率列
    prob_cols = [col for col in df_results.columns if col.startswith('prob_sol_')]

    # 设置matplotlib参数
    sns.set_style("whitegrid")
    
    for i, col in enumerate(prob_cols):
        solution_name = solution_maps.get(col, col)
        
        # 1. 箱线图/小提琴图：展示中位数和四分位数的趋势
        plt.figure(figsize=(10, 5))
        
        # 使用箱线图清晰展示统计分布
        sns.boxplot(
            data=df_results,
            x='cycle',
            y=col,
            order=['P4', 'P6', 'P8', 'P10'],
            palette="Set2"
        )
        # 添加中位数点的趋势线
        median_points = df_results.groupby('cycle')[col].median().reindex(['P4', 'P6', 'P8', 'P10'])
        plt.plot(median_points.index, median_points.values, marker='o', color='red', linestyle='--', label='Median Trend')
        
        plt.title(f'{solution_name} - Predicting the Probability of Aging Distribution (Box Plot)', fontsize=14)
        plt.xlabel('Cycle', fontsize=12)
        plt.ylabel('pred_possibility', fontsize=12)
        plt.ylim(-0.05, 1.05)
        plt.savefig(f'{col}_boxplot.png', dpi=300)
        plt.close()

        # 2. 核密度估计图 (KDE)：展示概率密度的重叠与分离
        plt.figure(figsize=(10, 5))
        
        # 绘制每个周期的概率密度
        for cycle in ['P4', 'P6', 'P8', 'P10']:
            subset = df_results[df_results['cycle'] == cycle]
            sns.kdeplot(
                subset[col],
                label=cycle,
                fill=True,
                alpha=0.4,
                linewidth=1.5
            )
            
        # 添加阈值线 (例如，0.5)
        plt.axvline(x=0.5, color='gray', linestyle='--', linewidth=1, label='Separation Threshold (0.5)')
        
        plt.title(f'{solution_name} - Predicting the Probability of Aging Dense (KDE Plot)', fontsize=14)
        plt.xlabel('pred_possibility', fontsize=12)
        plt.ylabel('dense', fontsize=12)
        plt.xlim(-0.05, 1.05)
        plt.legend(title='Cycle')
        plt.savefig(f'{col}_kdeplot.png', dpi=300)
        plt.close()
        
        print(f"-> Saved plots for {solution_name}")
# --- 主函数 ---
def main():
    # 1. 加载和准备数据
    df, X_all, y_soft, X_p4_p10, y_p4_p10, feature_cols = \
        load_and_prepare_data()
    
    input_dim = len(feature_cols)
    print(f"Model Input Dimension: {input_dim}")
    
    # 2. 运行四种方案
    df_res_1, _ = run_solution_1(df, X_all, X_p4_p10, y_p4_p10, input_dim)
    df_res_2, _ = run_solution_2(df, X_all, y_soft, input_dim)
    df_res_3, _ = run_solution_3(df, X_all)
    df_res_4, _, _ = run_solution_4(df, X_all, X_p4_p10, y_p4_p10, input_dim)
    
    # 3. 合并所有结果
    df_final = df.copy()
    df_final['prob_sol_1'] = df_res_1['prob_sol_1']
    df_final['prob_sol_2'] = df_res_2['prob_sol_2']
    df_final['prob_sol_3'] = df_res_3['prob_sol_3']
    df_final['prob_sol_4'] = df_res_4['prob_sol_4']
    
    # 4. 可视化
    plot_results(df_final)
    
    print("\nExperiment complete. Check 'solution_comparison_violin.png' for results.")

    # 5. !!! 单独效果可视化 !!!
    plot_individual_results(df_final)
    
    print("\nExperiment complete.")
    print("Please check the generated PNG files for all visualization results (comparison and individual).")

if __name__ == '__main__':
    main()