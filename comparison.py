import pandas as pd
import numpy as np
import torch
import joblib
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from scipy.stats import ttest_ind
import os
import sys

# 尝试导入你原有的模块
try:
    from upgrade import VAE, LatentRegressor
    from evaluation import transform_raw_data_to_model_input
    print("[1/5] 成功加载模型结构定义与预处理函数。")
except ImportError:
    print("错误: 找不到 upgrade.py 或 evaluation.py，请确保它们在当前目录。")
    sys.exit()

# =================================================================
# 1. 路径配置
# =================================================================
RAW_DATA_PATH = os.path.join('data','new_p6_data.csv' )
TRAIN_DATA_PATH = os.path.join('data', 'agg.csv')
META_FILE_PATH = 'inference_meta.json'
VAE_MODEL_PATH = 'best_vae_model.pth'
REGRESSOR_MODEL_PATH = 'best_regressor_model.pth'
SCALER_PATH = 'scaler.pkl'
OUTPUT_DIR = 'result' 
os.makedirs(OUTPUT_DIR, exist_ok=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def run_pca_comparison():
    print(f"[2/5] 正在处理数据: {RAW_DATA_PATH}...")

    # 1. 加载模型、元数据和 Scaler
    with open(META_FILE_PATH, 'r') as f:
        meta = json.load(f)
    scaler = joblib.load(SCALER_PATH)
    
    # 【终极防错机制】：直接从 scaler 中提取训练时的绝对特征名与顺序！
    if hasattr(scaler, 'feature_names_in_'):
        exact_features = list(scaler.feature_names_in_)
    else:
        # 兼容旧版本 sklearn (根据 upgrade.py 循环推导)
        exact_features = [f"{name}_{stat}" for stat in ['mean', 'median', 'std'] 
                          for name in ['adhesion', 'elastic_modulus', 'height', 'roughness', 'length', 'wideth']]
    
    # 2. 读取并预修复新数据 (大小写、拼写兼容)
    df_raw = pd.read_csv(RAW_DATA_PATH)
    current_cols_lower = {c.lower(): c for c in df_raw.columns}
    
    if 'condition' in current_cols_lower:
        df_raw.rename(columns={current_cols_lower['condition']: 'Condition'}, inplace=True)
    if 'wideth' in current_cols_lower:
        # 如果新数据自带错误拼写，先转回正确的以便 evaluation.py 清洗
        df_raw.rename(columns={current_cols_lower['wideth']: 'width'}, inplace=True)

    # 3. 数据清洗 (调用 evaluation.py 的原生函数)
    FEATURE_NAMES_CLEAN = ['adhesion', 'elastic_modulus', 'height', 'roughness', 'length', 'width']
    df_input = transform_raw_data_to_model_input(df_raw, FEATURE_NAMES_CLEAN)
    
    # 清洗完后，将 width_xxx 映射回带错别字的 wideth_xxx
    rename_map = {
        'width_mean': 'wideth_mean',
        'width_median': 'wideth_median',
        'width_std': 'wideth_std'
    }
    df_input.rename(columns=rename_map, inplace=True)
    
    # 严格按照 exact_features 提取并排序，杜绝一切 KeyError 和 ValueError
    X_new_scaled = scaler.transform(df_input[exact_features])
    print(f"[3/5] 数据预处理完成。成功对齐 {len(exact_features)} 个特征。样本量: {len(df_input)}")

    # 4. VAE 预测分支
    vae = VAE(input_dim=len(exact_features), latent_dim=meta['latent_dim']).to(device)
    vae.load_state_dict(torch.load(VAE_MODEL_PATH, map_location=device))
    vae.eval()

    regressor = LatentRegressor(input_dim=meta['latent_dim']).to(device)
    regressor.load_state_dict(torch.load(REGRESSOR_MODEL_PATH, map_location=device))
    regressor.eval()

    X_tensor = torch.tensor(X_new_scaled, dtype=torch.float32).to(device)
    with torch.no_grad():
        _, mu, _, _ = vae(X_tensor)
        mu_np = mu.cpu().numpy()
        # Z-Score 标准化
        mu_norm = (mu_np - mu_np.mean(axis=0)) / (mu_np.std(axis=0) + 1e-8)
        probs_vae = regressor(torch.tensor(mu_norm, dtype=torch.float32).to(device)).cpu().numpy().flatten()

    # 5. PCA 基准分支
    print("[4/5] 正在构建 PCA 基准对比...")
    df_train = pd.read_csv(TRAIN_DATA_PATH)
    soft_label_map = {'P4': 0.1, 'P6': 0.2, 'P8': 0.5, 'P10': 0.9}
    y_train = df_train['cycle'].map(soft_label_map).values
    
    # 同样对训练集做拼写替换
    df_train.rename(columns=rename_map, inplace=True, errors='ignore')
    X_train_scaled = scaler.transform(df_train[exact_features])

    pca = PCA(n_components=2)
    X_train_pca = pca.fit_transform(X_train_scaled)
    pca_reg = LinearRegression().fit(X_train_pca, y_train)

    X_new_pca = pca.transform(X_new_scaled)
    probs_pca = np.clip(pca_reg.predict(X_new_pca), 0, 1)

    # 6. 结果整合与可视化
    df_input['Prob_VAE'] = probs_vae
    df_input['Prob_PCA'] = probs_pca

    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    order = ['Normoxia', 'Hypoxia']
    
    sns.boxplot(ax=axes[0], x='Condition', y='Prob_VAE', data=df_input, order=order, palette='coolwarm')
    axes[0].set_title('VAE Model (Non-linear Latent Space)')
    axes[0].set_ylabel('Predicted Senescence Probability')

    sns.boxplot(ax=axes[1], x='Condition', y='Prob_PCA', data=df_input, order=order, palette='Greys')
    axes[1].set_title('PCA Baseline (Linear Projection)')
    axes[1].set_ylabel('Predicted Senescence Probability')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'VAE_vs_PCA_Comparison.pdf'), format='pdf')
    print("[5/5] 对比图表已保存至 VAE_vs_PCA_Comparison.pdf")

    # 7. 统计输出
    p_vae = ttest_ind(df_input[df_input['Condition']=='Normoxia']['Prob_VAE'],
                      df_input[df_input['Condition']=='Hypoxia']['Prob_VAE']).pvalue
    p_pca = ttest_ind(df_input[df_input['Condition']=='Normoxia']['Prob_PCA'],
                      df_input[df_input['Condition']=='Hypoxia']['Prob_PCA']).pvalue

    print("\n" + "="*55)
    print(f"{'Method':<15} | {'Normoxia Mean':<15} | {'Hypoxia Mean':<10} | {'P-value':<10}")
    print("-" * 55)
    print(f"{'VAE (Ours)':<15} | {df_input[df_input['Condition']=='Normoxia']['Prob_VAE'].mean():<15.4f} | "
          f"{df_input[df_input['Condition']=='Hypoxia']['Prob_VAE'].mean():<10.4f} | {p_vae:.2e}")
    print(f"{'PCA Baseline':<15} | {df_input[df_input['Condition']=='Normoxia']['Prob_PCA'].mean():<15.4f} | "
          f"{df_input[df_input['Condition']=='Hypoxia']['Prob_PCA'].mean():<10.4f} | {p_pca:.2e}")
    print("="*55)

if __name__ == "__main__":
    run_pca_comparison()