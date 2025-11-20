import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import seaborn as sns
import copy
import joblib  # 新增：用于保存 scaler
import json    # 新增：用于保存元数据
from typing import Tuple, List, Dict
import os
# --- 配置 ---
data_folder = 'data'
solution_folder = 'Solution_Path'
csv_filename = f"agg.csv"
FILE_PATH = os.path.join(data_folder, csv_filename)

LATENT_DIM = 2
RANDOM_SEED = 42

# 检查GPU可用性
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 设置随机种子以保证可复现性
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 150

# --- 1. 辅助类：早停机制 (Early Stopping) ---
class EarlyStopping:
    def __init__(self, patience=20, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta   
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

# --- 2. 数据加载与准备 ---
def load_and_prepare_data(filepath: str) -> Tuple[pd.DataFrame, torch.Tensor, torch.Tensor, list, StandardScaler]:
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        raise FileNotFoundError(f"Error: File '{filepath}' not found.")
    
    print(f"Loaded aggregated data with {len(df)} cells.")
    
    ignore_cols = ['cell_id', 'cycle']
    feature_cols = [col for col in df.columns if col not in ignore_cols]
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[feature_cols])
    
    # 使用非线性加速软标签
    soft_label_map = {'P4': 0.1, 'P6': 0.2, 'P8': 0.5, 'P10': 0.9}
    df['soft_label'] = df['cycle'].map(soft_label_map)

    X_all = torch.tensor(X_scaled, dtype=torch.float32)
    y_soft = torch.tensor(df['soft_label'].values, dtype=torch.float32)

    return df, X_all, y_soft, feature_cols, scaler

# --- 3. VAE 模型 (含 Dropout) ---
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

def vae_loss_function(recon_x, x, mu, logvar):
    recon_loss = nn.functional.mse_loss(recon_x, x, reduction='sum')
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kld

# --- 4. 回归器模型 ---
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

# --- 5. 自动调参训练函数 ---
def tune_and_train_best_vae(X_all: torch.Tensor, input_dim: int, latent_dim: int) -> VAE:
    print("\n--- Phase 1: Hyperparameter Tuning & Training VAE ---")
    param_grid = {
        'batch_size': [8,16,24,32],
        'lr': [5e-3,1e-3, 5e-4,1e-4],
        'dropout': [0.1,0.15, 0.2,0.25,0.3],
        'epochs': [500,600,700,800,900,1000]
    }
    
    dataset = TensorDataset(X_all)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(RANDOM_SEED))
    
    best_val_loss = float('inf')
    best_model_state = None
    best_params = {}
    
    for batch_size in param_grid['batch_size']:
        for lr in param_grid['lr']:
            for dropout in param_grid['dropout']:
                print(f"Testing: Batch={batch_size}, LR={lr}, Dropout={dropout}...", end=" ")
                
                train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
                
                model = VAE(input_dim, latent_dim=latent_dim, dropout_rate=dropout).to(device)
                optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
                early_stopping = EarlyStopping(patience=30, min_delta=0.1)
                
                epochs = param_grid['epochs'][0]
                
                for epoch in range(epochs):
                    model.train()
                    for batch in train_loader:
                        x_batch = batch[0].to(device)
                        optimizer.zero_grad()
                        recon_x, mu, logvar, _ = model(x_batch)
                        loss = vae_loss_function(recon_x, x_batch, mu, logvar)
                        loss.backward()
                        optimizer.step()
                    
                    model.eval()
                    val_loss_sum = 0
                    with torch.no_grad():
                        for batch in val_loader:
                            x_batch = batch[0].to(device)
                            recon_x, mu, logvar, _ = model(x_batch)
                            val_loss_sum += vae_loss_function(recon_x, x_batch, mu, logvar).item()
                    
                    avg_val_loss = val_loss_sum / len(val_loader.dataset)
                    
                    early_stopping(avg_val_loss)
                    if early_stopping.early_stop:
                        break
                
                print(f"Final Val Loss: {avg_val_loss:.4f}")
                
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    best_model_state = copy.deepcopy(model.state_dict())
                    best_params = {'batch_size': batch_size, 'lr': lr, 'dropout': dropout}

    print(f"\n>>> Best VAE Params: {best_params} | Best Val Loss: {best_val_loss:.4f}")
    final_model = VAE(input_dim, latent_dim=latent_dim, dropout_rate=best_params['dropout']).to(device)
    final_model.load_state_dict(best_model_state)
    final_model.eval()
    return final_model

def train_latent_regressor(latent_dim: int, Z_all: torch.Tensor, y_soft: torch.Tensor) -> LatentRegressor:
    print("\n--- Phase 2: Training Latent Regressor ---")
    dataset = TensorDataset(Z_all, y_soft)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(RANDOM_SEED))
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    
    model = LatentRegressor(input_dim=latent_dim, dropout_rate=0.1).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    early_stopping = EarlyStopping(patience=20, min_delta=0.001)
    
    for epoch in range(300):
        model.train()
        for z_batch, y_batch in train_loader:
            z_batch, y_batch = z_batch.to(device), y_batch.to(device).view(-1, 1)
            optimizer.zero_grad()
            probs = model(z_batch)
            loss = criterion(probs, y_batch)
            loss.backward()
            optimizer.step()
            
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for z_batch, y_batch in val_loader:
                z_batch, y_batch = z_batch.to(device), y_batch.to(device).view(-1, 1)
                val_loss += criterion(model(z_batch), y_batch).item()
        
        avg_val_loss = val_loss / len(val_loader)
        early_stopping(avg_val_loss)
        if early_stopping.early_stop:
            print(f"Regressor Early stopping at epoch {epoch+1}")
            break
            
    return model

# --- 6. 轴反转校准 ---
def align_aging_axis(df_latent: pd.DataFrame, latent_col: str = 'z0', score_col: str = 'prob_sol_4B') -> Tuple[pd.DataFrame, float]:
    print("\n--- Performing Axis Calibration ---")
    correlation, _ = spearmanr(df_latent[latent_col], df_latent[score_col])
    df_calibrated = df_latent.copy()
    flip_factor = 1.0
    if correlation < 0:
        print(f"(!) Axis Inversion Detected (Corr={correlation:.3f}). Flipping {latent_col} axis.")
        df_calibrated[f'{latent_col}_aligned'] = -df_latent[latent_col]
        flip_factor = -1.0
    else:
        print(f"(v) Axis Correct (Corr={correlation:.3f}). Keeping {latent_col} axis.")
        df_calibrated[f'{latent_col}_aligned'] = df_latent[latent_col]
    return df_calibrated, flip_factor

# --- 7. 模型保存机制 (新增) ---
def save_full_model_pipeline(vae_model, regressor_model, scaler, flip_factor, feature_cols, latent_dim):
    """保存模型、Scaler和必要的元数据"""
    print("\n--- Saving Model Pipeline ---")
    
    # 1. 保存模型权重
    torch.save(vae_model.state_dict(), 'best_vae_model.pth')
    torch.save(regressor_model.state_dict(), 'best_regressor_model.pth')
    print("Saved 'best_vae_model.pth' and 'best_regressor_model.pth'")
    
    # 2. 保存 Scaler (使用 joblib)
    joblib.dump(scaler, 'scaler.pkl')
    print("Saved 'scaler.pkl'")
    
    # 3. 保存推理元数据 (JSON)
    meta_data = {
        'feature_cols': feature_cols,
        'latent_dim': latent_dim,
        'flip_factor': flip_factor,  # 关键：推理时必须乘以这个因子
        'input_dim': len(feature_cols)
    }
    with open('inference_meta.json', 'w') as f:
        json.dump(meta_data, f, indent=4)
    print("Saved 'inference_meta.json'")

# --- 8. 绘图函数 ---
def plot_continuous_latent_space(df_latent: pd.DataFrame, x_col: str):
    print(f"Generating calibrated latent space plot using x-axis: {x_col}...")
    plt.figure(figsize=(10, 7))
    sc = plt.scatter(x=df_latent[x_col], y=df_latent['z1'], c=df_latent['prob_sol_4B'], cmap='coolwarm', s=50, alpha=0.8)
    cbar = plt.colorbar(sc)
    cbar.set_label("Predicted Senescence Probability", rotation=270, labelpad=15, fontsize=12)
    plt.title('VAE Latent Space (Calibrated) - Colored by Probability', fontsize=16)
    plt.xlabel(f'Latent Dimension 1 ({x_col}) [Low->High Aging]', fontsize=12)
    plt.ylabel('Latent Dimension 2 (z1)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig('optimized_latent_space_probability.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_final_distributions(df_results: pd.DataFrame):
    print("Generating distribution plots...")
    col = 'prob_sol_4B'
    plt.figure(figsize=(10, 5))
    sns.boxplot(data=df_results, x='cycle', y=col, order=['P4', 'P6', 'P8', 'P10'], palette="Set2")
    plt.title('VAE + Regressor (Optimized) - Probability Distribution', fontsize=14)
    plt.ylim(-0.05, 1.05)
    plt.savefig('optimized_boxplot.png', dpi=300)
    plt.close()

    plt.figure(figsize=(10, 5))
    for cycle in ['P4', 'P6', 'P8', 'P10']:
        sns.kdeplot(df_results[df_results['cycle'] == cycle][col], label=cycle, fill=True, alpha=0.4, linewidth=1.5)
    plt.title('VAE + Regressor (Optimized) - Probability Density', fontsize=14)
    plt.xlim(-0.05, 1.05)
    plt.legend(title='Cycle')
    plt.savefig('optimized_kdeplot.png', dpi=300)
    plt.close()

def analyze_latent_correlations_clean(df: pd.DataFrame, feature_cols: list, z0_col: str = 'z0_aligned'):
    print("\n[Method 1] Generating cleaned correlation heatmap (No STD)...")
    clean_features = [f for f in feature_cols if 'std' not in f]
    df_corr_input = pd.concat([df[z0_col], df['z1'], df[clean_features]], axis=1)
    corr_matrix = df_corr_input.corr(method='spearman')
    z_corr = corr_matrix.loc[clean_features, [z0_col, 'z1']]
    plt.figure(figsize=(8, 10))
    sns.heatmap(z_corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0, fmt='.2f')
    plt.title('Correlation: Latent Dims vs Features (Mean/Median Only)')
    plt.tight_layout()
    plt.savefig('optimized_analysis_1_heatmap.png', dpi=300)
    plt.close()

def plot_feature_overlay_calibrated(df: pd.DataFrame, feature_cols: list, z0_col: str = 'z0_aligned'):
    print("\n[Method 2] Generating Feature Overlay Plots...")
    key_features = [f for f in feature_cols if 'mean' in f][:4] 
    for feature in key_features:
        if feature not in df.columns: continue
        plt.figure(figsize=(10, 7))
        sc = plt.scatter(x=df[z0_col], y=df['z1'], c=df[feature], cmap='viridis', s=50, alpha=0.8)
        cbar = plt.colorbar(sc)
        cbar.set_label(feature, rotation=270, labelpad=15, fontsize=12)
        plt.title(f'Latent Space colored by {feature}', fontsize=16)
        plt.xlabel(f'Latent Dimension 1 ({z0_col})', fontsize=12)
        plt.ylabel('Latent Dimension 2 (z1)', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.savefig(f'optimized_analysis_2_map_{feature}.png', dpi=300, bbox_inches='tight')
        plt.close()

def analyze_decoder_traversal_calibrated(model_vae: VAE, scaler: StandardScaler, feature_cols: List[str], latent_dim: int, flip_factor: float):
    print("\n[Method 3] Running Decoder Traversal Analysis...")
    model_vae.eval()
    visual_steps = np.linspace(-3, 3, 20)
    z_traversal = np.zeros((20, latent_dim))
    z_traversal[:, 0] = visual_steps * flip_factor
    z_tensor = torch.tensor(z_traversal, dtype=torch.float32).to(device)
    with torch.no_grad():
        recon_x_scaled = model_vae.decoder(z_tensor).cpu().numpy()
    recon_x_original = scaler.inverse_transform(recon_x_scaled)
    df_recon = pd.DataFrame(recon_x_original, columns=feature_cols)
    df_recon['Aging_Axis'] = visual_steps
    cols_to_plot = [c for c in feature_cols if 'mean' in c]
    if not cols_to_plot: cols_to_plot = feature_cols[:6]

    plt.figure(figsize=(12, 7))
    for col in cols_to_plot:
        y_vals = df_recon[col]
        y_norm = (y_vals - y_vals.min()) / (y_vals.max() - y_vals.min())
        plt.plot(df_recon['Aging_Axis'], y_norm, label=col, linewidth=2, marker='o', markersize=4)
    plt.title('Reconstructed Features along Aligned Aging Axis', fontsize=16)
    plt.xlabel('Latent Dimension z0 (<- Young | Old ->)', fontsize=12)
    plt.ylabel('Normalized Feature Value (0-1)', fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2)
    plt.grid(True, linestyle='--')
    plt.tight_layout()
    plt.savefig('optimized_analysis_3_traversal.png', dpi=300)
    plt.close()

# --- 9. 主执行流程 ---
if __name__ == '__main__':
    # 1. 加载数据
    df, X_all, y_soft, feature_cols, scaler = load_and_prepare_data(FILE_PATH)
    INPUT_DIM = len(feature_cols)
    
    # 2. 阶段一：自动调参并训练最佳 VAE
    model_vae = tune_and_train_best_vae(X_all, INPUT_DIM, LATENT_DIM)
    
    # 3. 提取潜在空间
    with torch.no_grad():
        _, mu_all, _, _ = model_vae(X_all.to(device))
    z_mu_all_np = mu_all.cpu().numpy()
    
    df_latent = pd.DataFrame(z_mu_all_np, columns=['z0', 'z1'])
    df_latent['cycle'] = df['cycle']
    
    # 4. 阶段二：训练回归器
    Z_all_tensor = torch.tensor(z_mu_all_np, dtype=torch.float32)
    model_regressor = train_latent_regressor(LATENT_DIM, Z_all_tensor, y_soft)
    
    # 5. 获取最终预测
    with torch.no_grad():
        final_probs = model_regressor(Z_all_tensor.to(device)).cpu().numpy().flatten()
    df['prob_sol_4B'] = final_probs
    df_latent['prob_sol_4B'] = final_probs
    
    # 6. 轴反转校准
    df_latent_calibrated, flip_factor = align_aging_axis(df_latent, latent_col='z0', score_col='prob_sol_4B')
    aligned_z0_col = 'z0_aligned'
    df[aligned_z0_col] = df_latent_calibrated[aligned_z0_col]
    df['z1'] = df_latent_calibrated['z1']
    
    # 7. 保存模型和元数据 (新增)
    save_full_model_pipeline(model_vae, model_regressor, scaler, flip_factor, feature_cols, LATENT_DIM)
    
    # 8. 生成图表
    plot_continuous_latent_space(df, x_col=aligned_z0_col)
    plot_final_distributions(df)
    analyze_latent_correlations_clean(df, feature_cols, z0_col=aligned_z0_col)
    plot_feature_overlay_calibrated(df, feature_cols, z0_col=aligned_z0_col)
    analyze_decoder_traversal_calibrated(model_vae, scaler, feature_cols, LATENT_DIM, flip_factor)
    
    print("\nOptimization Complete. Models saved and analyses generated.")