import torch
import numpy as np
import pandas as pd
import joblib
import json
import os
import sys
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, confusion_matrix, accuracy_score
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression

# 尝试导入模型定义，如果导入失败则报错
try:
    from upgrade import VAE, LatentRegressor
    print("[1/6] load model definitions successfully.")
except ImportError:
    print("错误: 无法找到 upgrade.py。请确保此脚本与训练脚本在同一目录。")
    sys.exit()

# --- 配置 ---
CONFIG = {
    "RANDOM_SEED": 42,
    "META_PATH": 'inference_meta.json',
    "SCALER_PATH": 'scaler.pkl',
    "VAE_WEIGHTS": 'best_vae_model.pth',
    "REG_WEIGHTS": 'best_regressor_model.pth',
    "DATA_PATH": os.path.join('data', 'agg.csv')
}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def evaluate():
    print(f"[2/6] Using device: {device}")
    
    # 1. 检查文件是否存在
    for key, path in CONFIG.items():
        if key.endswith('PATH') or key.endswith('WEIGHTS'):
            if not os.path.exists(path):
                print(f"缺失文件: {path}。请检查该文件是否在当前目录。")
                return

    # 2. 加载元数据和数据
    try:
        with open(CONFIG["META_PATH"], 'r') as f:
            meta = json.load(f)
        feature_cols = meta['feature_cols']
        latent_dim = meta['latent_dim']
        scaler = joblib.load(CONFIG["SCALER_PATH"])
        print("[3/6] load metadata and scaler successfully.")
    except Exception as e:
        print(f"加载元数据失败: {e}")
        return

    # 3. 数据准备 (严格复现 upgrade.py 逻辑)
    df = pd.read_csv(CONFIG["DATA_PATH"])
    soft_label_map = {'P4': 0.1, 'P6': 0.2, 'P8': 0.5, 'P10': 0.9}
    y_true = df['cycle'].map(soft_label_map).values
    X_scaled = scaler.transform(df[feature_cols])

    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    y_tensor = torch.tensor(y_true, dtype=torch.float32)
    dataset = TensorDataset(X_tensor, y_tensor)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    # 使用固定种子复现划分
    _, val_dataset = random_split(
        dataset, [train_size, val_size], 
        generator=torch.Generator().manual_seed(CONFIG["RANDOM_SEED"])
    )
    print(f"[4/6] Split datasets: {len(val_dataset)}")

    # 4. 加载权重
    try:
        vae = VAE(input_dim=len(feature_cols), latent_dim=latent_dim).to(device)
        vae.load_state_dict(torch.load(CONFIG["VAE_WEIGHTS"], map_location=device))
        vae.eval()

        regressor = LatentRegressor(input_dim=latent_dim).to(device)
        regressor.load_state_dict(torch.load(CONFIG["REG_WEIGHTS"], map_location=device))
        regressor.eval()
        print("[5/6] load model weights successfully.")
    except Exception as e:
        print(f"加载模型权重失败: {e}")
        return

    # 5. 执行评估
    print("[6/6] calculating metrics on validation set...")
    loader = DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=False)
    
    with torch.no_grad():
        x_batch, y_batch = next(iter(loader))
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        
        # VAE 重构指标
        recon_x, mu, _, _ = vae(x_batch)
        recon_mse = torch.nn.functional.mse_loss(recon_x, x_batch).item()
        
        # 回归预测指标
        probs = regressor(mu).cpu().numpy().flatten()
        y_true_np = y_batch.cpu().numpy()

    # 指标计算
    r2 = r2_score(y_true_np, probs)
    mse = mean_squared_error(y_true_np, probs)
    mae = mean_absolute_error(y_true_np, probs)

    # 分类转换 (阈值 0.5)
    y_class_true = (y_true_np >= 0.5).astype(int)
    y_class_pred = (probs >= 0.5).astype(int)
    acc = accuracy_score(y_class_true, y_class_pred)
    tn, fp, fn, tp = confusion_matrix(y_class_true, y_class_pred).ravel()
    
    # 预防除零错误
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0

    # 6. PCA 基准对比
    pca = PCA(n_components=2)
    # 模拟训练 PCA (使用全体数据以获取基准方向)
    pca.fit(X_scaled) 
    X_val_pca = pca.transform(X_tensor[val_dataset.indices].numpy())
    y_val_pca = y_tensor[val_dataset.indices].numpy()
    
    # 简单的线性回归作为对比
    pca_reg = LinearRegression().fit(X_val_pca, y_val_pca) # 仅用验证集拟合一个最简单的线性模型
    pca_r2 = pca_reg.score(X_val_pca, y_val_pca)

    # --- 输出结果 ---
    print("\n" + "="*60)
    print("             VAE + MLP model quantitative measure  (validation set)             ")
    print("="*60)
    metrics_table = {
        "Reconstruction MSE": f"{recon_mse:.6f}",
        "Regression R^2": f"{r2:.4f}",
        "Regression MSE": f"{mse:.4f}",
        "Regression MAE": f"{mae:.4f}",
        "Classification Accuracy": f"{acc:.2%}",
        "Sensitivity (Recall)": f"{sens:.2%}",
        "Specificity": f"{spec:.2%}",
        "PCA Baseline R^2": f"{pca_r2:.4f}"
    }
    
    for k, v in metrics_table.items():
        print(f"{k:<35}: {v}")
    
    improvement = (r2 - pca_r2) / abs(pca_r2) if pca_r2 != 0 else 0
    print("-" * 60)
    print(f"result:VAE peform better than PCA: {improvement:.2%}")
    print("="*60)

if __name__ == '__main__':
    evaluate()