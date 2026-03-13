import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import joblib
import os

# ==========================================
# 1. 路径与特征配置
# ==========================================
DATA_PATH = 'data/agg.csv'  # 已构造好 18 维特征的文件
SCALER_PATH = 'scaler.pkl'   # 训练时用的 scaler
OUTPUT_DIR = 'result' 
os.makedirs(OUTPUT_DIR, exist_ok=True)
# 明确 18 个特征的名称
BASE_FEATURES = ['adhesion', 'elastic_modulus', 'height', 'roughness', 'length', 'wideth']
STATS = ['mean', 'median', 'std']
MODEL_INPUT_FEATURES = [f'{name}_{stat}' for stat in STATS for name in BASE_FEATURES]

# ==========================================
# 2. 加载数据与标准化
# ==========================================
if not os.path.exists(DATA_PATH):
    print(f"错误: 找不到文件 {DATA_PATH}，请确认数据已放在 data 文件夹下。")
    exit()

df = pd.read_csv(DATA_PATH)
X = df[MODEL_INPUT_FEATURES].values
labels = df['cycle'].values
order = ['P4', 'P6', 'P8', 'P10']

# 加载标准化器
if os.path.exists(SCALER_PATH):
    scaler = joblib.load(SCALER_PATH)
    X_scaled = scaler.transform(X)
else:
    from sklearn.preprocessing import StandardScaler
    X_scaled = StandardScaler().fit_transform(X)
    print("提示: 未找到 scaler.pkl，已对当前数据进行实时标准化。")

# ==========================================
# 3. 执行 PCA 降维
# ==========================================
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# 获取各主成分的贡献率
var_exp = pca.explained_variance_ratio_

# ==========================================
# 4. 绘图
# ==========================================
plt.figure(figsize=(10, 7))
sns.set_style("whitegrid")

# 使用渐变色板，体现衰老演进
scatter = sns.scatterplot(
    x=X_pca[:, 0], 
    y=X_pca[:, 1], 
    hue=labels, 
    hue_order=order,
    palette="magma", 
    s=100, 
    alpha=0.75, 
    edgecolor='k', 
    linewidth=0.8
)

# 动态设置标题，包含解释方差信息
plt.title(f"PCA of Physical Features\nTotal Explained Variance: {var_exp.sum():.2%}", 
          fontsize=14, fontweight='bold')
plt.xlabel(f"PC1 ({var_exp[0]:.1%})", fontsize=12)
plt.ylabel(f"PC2 ({var_exp[1]:.1%})", fontsize=12)

plt.legend(title="Cell Cycle", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, linestyle='--', alpha=0.5)

# 保存为 PDF
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'original_data_pca_analysis.pdf'), format='pdf')
print("PCA 绘图已完成并保存为 'original_data_pca_analysis.pdf'")
plt.show()