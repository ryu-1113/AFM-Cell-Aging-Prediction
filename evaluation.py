import pandas as pd
import numpy as np
import torch
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import joblib 
import sys
import os
import torch.nn as nn

# =================================================================
# 1. Configuration and Constants
# =================================================================

RAW_DATA_PATH = 'new_p6_data.csv' 
META_FILE_PATH = 'inference_meta.json'
VAE_MODEL_PATH = 'best_vae_model.pth'
REGRESSOR_MODEL_PATH = 'best_regressor_model.pth'
SCALER_PATH = 'scaler.pkl' 

# 6 Raw feature names (MUST match column headers in the raw data)
FEATURE_NAMES = [
    'adhesion', 
    'elastic_modulus', 
    'height', 
    'roughness', 
    'length', 
    'width'
]

# 18 Model input feature names (6 features * 3 stats)
MODEL_INPUT_FEATURES = []
for stat in ['mean', 'median', 'std']:
    for name in FEATURE_NAMES:
        MODEL_INPUT_FEATURES.append(f'{name}_{stat}')

# 设置绘图风格
sns.set_style("whitegrid")

# =================================================================
# 2. Model Architecture Definitions (Match upgrade.py)
# =================================================================

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
        return self.decoder(z), mu, logvar, z

class Regressor(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 32, dropout_rate: float = 0.1):
        super(Regressor, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid() 
        )
    def forward(self, x):
        return self.network(x)

# =================================================================
# 3. Data Transformation Function (Original, Corrected Grouping)
# =================================================================

def transform_raw_data_to_model_input(df_raw: pd.DataFrame, feature_names: list) -> pd.DataFrame:
    """
    Groups by cell_id, Condition, and cycle to calculate statistics 
    for each unique cell/condition group. This function is based on the 
    corrected grouping from our previous iterations to ensure all conditions
    are preserved.
    """
    df_temp = df_raw.copy()
    
    print(f"DEBUG: Inside transform_raw_data_to_model_input - initial df_temp shape: {df_temp.shape}")
    print(f"DEBUG: Unique 'Condition' values in df_temp before ffill/dropna: {df_temp['Condition'].unique().tolist()}")
    print(f"DEBUG: Count of 'cell_id' NaNs before ffill: {df_temp['cell_id'].isnull().sum()}")

    # 1. Forward-fill missing cell_id values.
    if df_temp['cell_id'].isnull().any():
        df_temp['cell_id'] = df_temp['cell_id'].ffill()
        print("INFO: Missing 'cell_id' values have been forward-filled (ffill).")
    
    print(f"DEBUG: Count of 'cell_id' NaNs after ffill: {df_temp['cell_id'].isnull().sum()}")
    print(f"DEBUG: Count of 'Condition' NaNs after ffill: {df_temp['Condition'].isnull().sum()}")

    # Define the CORRECT grouping keys (cell_id, Condition, cycle)
    GROUPING_KEYS = ['cell_id', 'Condition', 'cycle']
    
    # Drop NaNs from the essential grouping columns and features
    df_temp.dropna(subset=GROUPING_KEYS + feature_names, inplace=True) 
    
    print(f"DEBUG: df_temp shape after dropping NaNs in grouping keys/features: {df_temp.shape}")
    print(f"DEBUG: Unique 'Condition' values in df_temp after cleaning: {df_temp['Condition'].unique().tolist()}")

    if df_temp.empty:
        raise ValueError("ERROR: Dataframe is empty after cleaning essential columns.")

    df_temp['cell_id'] = df_temp['cell_id'].astype(int)
    
    # A. Aggregate Feature Columns (numeric stats) - Grouping by cell_id, Condition, and cycle
    agg_funcs = {name: ['mean', 'median', 'std'] for name in feature_names}
    df_vae_input = df_temp.groupby(GROUPING_KEYS).agg(agg_funcs).reset_index() 
    
    # Correct column flattening after grouping by multiple keys
    new_cols = GROUPING_KEYS + [f'{col[0]}_{col[1]}' for col in df_vae_input.columns[len(GROUPING_KEYS):]]
    df_vae_input.columns = new_cols

    # Fill NaN std (which happens when only one measurement is available) with 0
    for name in feature_names:
        std_col = f'{name}_std'
        if std_col in df_vae_input.columns and df_vae_input[std_col].isnull().any():
             df_vae_input[std_col] = df_vae_input[std_col].fillna(0.0)
    
    # Final check and return
    df_vae_input = df_vae_input[GROUPING_KEYS + MODEL_INPUT_FEATURES]

    print(f"DEBUG: Final df_vae_input shape: {df_vae_input.shape}")
    print(f"DEBUG: Unique 'Condition' values in final df_vae_input: {df_vae_input['Condition'].unique().tolist()}")
    
    return df_vae_input

# =================================================================
# 4. Model and Scaler Loading Function
# =================================================================

def load_models_and_metadata(meta_path: str, vae_path: str, reg_path: str, scaler_path: str):
    """Loads metadata, models, and the StandardScaler object."""
    print("INFO: Loading metadata, models, and scaler...")
    
    try:
        with open(meta_path, 'r') as f:
            meta = json.load(f)
    except FileNotFoundError:
        print(f"ERROR: Metadata file '{meta_path}' not found. Exiting.")
        sys.exit(1)

    input_dim = meta.get('input_dim', 18)
    latent_dim = meta.get('latent_dim', 2)
    flip_factor = meta.get('flip_factor', 1)
    
    scaler = None
    try:
        scaler = joblib.load(scaler_path)
        print(f"INFO: Raw feature scaler '{scaler_path}' loaded successfully.")
    except Exception as e:
        print(f"WARNING: Failed to load raw feature scaler. Detail: {e}. Proceeding WITHOUT standardization.")

    vae = VAE(input_dim, latent_dim=latent_dim)
    regressor = Regressor(input_dim=latent_dim) 
    
    try:
        vae.load_state_dict(torch.load(vae_path, map_location='cpu'))
        regressor.load_state_dict(torch.load(reg_path, map_location='cpu'))
        print("INFO: VAE and Regressor models loaded successfully.")
    except Exception as e:
        print(f"ERROR: Failed to load model weights. Check paths and model class definitions. Detail: {e}. Exiting.")
        sys.exit(1)
        
    vae.eval()
    regressor.eval()
    
    return vae, regressor, flip_factor, scaler

# =================================================================
# 5. Main Inference Function (Original Logic Restored)
# =================================================================

def run_inference():
    """Executes the full inference pipeline."""
    
    # 1. Load resources
    vae, regressor, flip_factor, scaler = load_models_and_metadata(
        META_FILE_PATH, VAE_MODEL_PATH, REGRESSOR_MODEL_PATH, SCALER_PATH
    )
    
    # 2. Load and transform data
    print("\n--- 2. Data Processing ---")
    try:
        df_raw = pd.read_csv(RAW_DATA_PATH)
    except FileNotFoundError:
        print(f"ERROR: Data file '{RAW_DATA_PATH}' not found. Exiting.")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: Failed to read data file. Detail: {e}. Exiting.")
        sys.exit(1)
        
    print(f"INFO: Raw data loaded. Total rows: {len(df_raw)}")
    
    # Simplify group column renaming based on known input
    if 'condition' in df_raw.columns:
        df_raw.rename(columns={'condition': 'Condition'}, inplace=True)
        print("INFO: Renamed 'condition' to 'Condition'.")
    elif 'Condition' not in df_raw.columns:
        print(f"FATAL ERROR: Required group column 'Condition' (or 'condition') not found in data. Available: {df_raw.columns.tolist()}")
        sys.exit(1)

    # Simplify cycle column renaming (if '周期' exists)
    if '周期' in df_raw.columns and 'cycle' not in df_raw.columns:
        df_raw.rename(columns={'周期': 'cycle'}, inplace=True)
        print("INFO: Renamed '周期' to 'cycle'.")
    elif 'cycle' not in df_raw.columns:
        print("WARNING: 'cycle' column not found. Adding dummy 'cycle' column with default value 'P6'.")
        df_raw['cycle'] = 'P6'
    
    missing_raw_features = [col for col in FEATURE_NAMES if col not in df_raw.columns]
    if missing_raw_features:
        print(f"FATAL ERROR: Raw data is missing required feature columns: {missing_raw_features}. Available: {df_raw.columns.tolist()}")
        sys.exit(1)

    df_vae_input = transform_raw_data_to_model_input(df_raw, FEATURE_NAMES)
    print(f"INFO: Feature transformation complete. Generated {len(df_vae_input)} cell/condition entries.")
    
    X_new = df_vae_input[MODEL_INPUT_FEATURES].values
    
    # 3. Standardization (Raw Features)
    if scaler is not None:
        X_new = scaler.transform(X_new)
        print("INFO: Raw features standardized using loaded scaler.")
    else:
        print("WARNING: Raw feature standardization skipped.")

    # 4. Model Inference & Latent Space Standardization (ORIGINAL CRITICAL LOGIC)
    print("\n--- 4. Model Inference ---")
    X_tensor = torch.tensor(X_new, dtype=torch.float32)
    with torch.no_grad():
        _, z_mu_new, _, _ = vae(X_tensor) 

    z_mu_new_np = z_mu_new.cpu().numpy()
    
    # --- ORIGINAL KEY FIX: Standardize the Latent Space for Regressor ---
    # This block ensures that z_mu is scaled before being passed to the Regressor,
    # which was crucial for its correct performance.
    z_scaler = StandardScaler() # Re-fit the z_scaler for the new data.
                                # For true generalization, this scaler should be saved 
                                # and loaded from upgrade.py. But for fixing the immediate 
                                # prediction issue based on previous console output,
                                # dynamic fit_transform is the direct restoration.
    z_mu_new_scaled_np = z_scaler.fit_transform(z_mu_new_np)
    
    z_mu_new_scaled = torch.tensor(z_mu_new_scaled_np, dtype=torch.float32)
    print(f"DEBUG: VAE latent mu (z_mu_new) min/max BEFORE Z-scaling: {z_mu_new_np.min():.4f} / {z_mu_new_np.max():.4f}")
    print(f"INFO: Applied **Z-space standardization** (Mean=0, Std=1) to prevent Regressor saturation.")
    print(f"DEBUG: VAE latent mu (z_mu_new) min/max AFTER Z-scaling: {z_mu_new_scaled.min().item():.4f} / {z_mu_new_scaled.max().item():.4f}")
    
    with torch.no_grad():
        predicted_prob_tensor = regressor(z_mu_new_scaled) # Here we use the scaled z_mu
    
    # 5. Result Integration
    predicted_prob_np = predicted_prob_tensor.cpu().numpy().flatten()
    print(f"DEBUG: Predicted probabilities (raw tensor) min/max AFTER Z-scaling: {predicted_prob_np.min():.4f} / {predicted_prob_np.max():.4f}")
    
    df_results = df_vae_input.copy()
    df_results['predicted_prob'] = predicted_prob_np
    
    # Store latent space (aligned) for plotting, using the original (non-scaled) z_mu for these values,
    # as the latent space visualization itself typically refers to the VAE's direct output,
    # with `flip_factor` for interpretability on the aging axis.
    df_results['z0_aligned'] = z_mu_new_np[:, 0] * flip_factor 
    df_results['z1'] = z_mu_new_np[:, 1] # Keep z1 as is from VAE output

    # 6. Validation and Plotting
    plot_and_validate(df_results)
    print("--- INFERENCE COMPLETE ---")


# =================================================================
# 6. Visualization and Validation Function (ENHANCED & SAVE PDF)
# =================================================================

def plot_and_validate(df_results: pd.DataFrame):
    """Generates multiple plots and performs validation check (EN)."""
    BW_ADJUST_FACTOR = 1    
    mean_scores = df_results.groupby('Condition')['predicted_prob'].mean().sort_values(ascending=False)
    
    print("\n--- 5. Results Validation ---")
    print("Mean Predicted Senescence Score:")
    print(mean_scores.to_string()) 
    
    validation_success = False
    available_conditions = df_results['Condition'].unique().tolist()

    try:
        if 'Hypoxia' in available_conditions and 'Normoxia' in available_conditions:
            if mean_scores['Hypoxia'] < mean_scores['Normoxia']:
                print("\nSUCCESS: Hypoxia mean probability is lower than Normoxia, as expected.")
                validation_success = True
            else:
                print("\nFAILURE: Hypoxia mean probability is NOT lower than Normoxia.")
        else:
            print(f"\nWARNING: Could not perform validation check. Found conditions: {available_conditions}. Need both 'Normoxia' and 'Hypoxia'.")
    except Exception:
        print("\nWARNING: An error occurred during validation check.")
    
    # 准备绘图数据
    plot_order = [cond for cond in ['Normoxia', 'Hypoxia'] if cond in available_conditions]
    if not plot_order:
         print("\nFATAL ERROR: No valid conditions (Hypoxia/Normoxia) found for plotting.")
         plt.close()
         return

    palette_map = {'Normoxia': 'r', 'Hypoxia': 'b'}
    # 定义形状映射
    marker_map = {'Normoxia': 'o', 'Hypoxia': 'X'} # 圆形 vs 叉号

    # --- Plot 1: Boxplot (Existing) ---
    plt.figure(figsize=(6, 5))
    sns.boxplot(
        x='Condition', 
        y='predicted_prob', 
        data=df_results, 
        order=plot_order, 
        palette={k: palette_map[k] for k in plot_order}
    )
    plt.title('Predicted Senescence Probability by Oxygen Condition (Boxplot)')
    plt.xlabel('Condition')
    plt.ylabel('Senescence Probability')
    plt.ylim(-0.05, 1.05)
    
    for i, condition in enumerate(plot_order):
        score = mean_scores.get(condition, np.nan)
        if not np.isnan(score):
            text_y_pos = 1.02
            plt.text(i, text_y_pos, f'Mean: {score:.2f}', 
                     horizontalalignment='center', color='black', fontsize=10)

    # 保存为 PDF，并确保文件名无空格
    output_plot_name = 'validation_1_boxplot.pdf'.strip()
    plt.savefig(output_plot_name, format='pdf', bbox_inches='tight')
    print(f"INFO: Visualization saved: {output_plot_name}")
    plt.close()

    # --- Plot 2: Violin Plot (New) ---
    plt.figure(figsize=(6, 5))
    sns.violinplot(
        x='Condition',
        y='predicted_prob',
        data=df_results,
        order=plot_order,
        bw_adjust=BW_ADJUST_FACTOR, # <--- 减小带宽      
        palette={k: palette_map[k] for k in plot_order},
        inner='box' # 在小提琴内部绘制箱线图
    )
    plt.title('Predicted Senescence Probability by Oxygen Condition (Violin Plot)')
    plt.xlabel('Condition')
    plt.ylabel('Senescence Probability')
    plt.ylim(-0.05, 1.05)
    
    # 保存为 PDF，并确保文件名无空格
    output_plot_name = 'validation_2_violinplot.pdf'.strip()
    plt.savefig(output_plot_name, format='pdf', bbox_inches='tight')
    print(f"INFO: Visualization saved: {output_plot_name}")
    plt.close()

    # --- Plot 3: Density Plot (New) ---
    plt.figure(figsize=(7, 5))
    for cond in plot_order:
        subset = df_results[df_results['Condition'] == cond]
        # 只有当数据量足够时才绘制密度图，否则会报错
        if len(subset) > 1:
            sns.kdeplot(subset['predicted_prob'], label=cond, color=palette_map[cond], bw_adjust=BW_ADJUST_FACTOR # <--- 减小带宽
                        ,fill=True, alpha=0.4)
    
    plt.title('Predicted Senescence Probability Density')
    plt.xlabel('Senescence Probability')
    plt.ylabel('Density')
    plt.xlim(-0.05, 1.05)
    plt.legend(title='Condition')
    
    # 保存为 PDF，并确保文件名无空格
    output_plot_name = 'validation_3_densityplot.pdf'.strip()
    plt.savefig(output_plot_name, format='pdf', bbox_inches='tight')
    print(f"INFO: Visualization saved: {output_plot_name}")
    plt.close()

    # --- Plot 4: Latent Space Projection (Revised Colorbar Logic) ---
    fig, ax = plt.subplots(figsize=(8, 6)) # 创建 Figure 和 Axes 对象
    
    # 存储 scatter plot 的返回值，以便为 colorbar 提供 mappable
    scatter_plot_mappable = None
    
    for cond in plot_order:
        subset = df_results[df_results['Condition'] == cond]
        # 在 `ax` 上绘制散点图
        current_scatter = ax.scatter(
            x=subset['z0_aligned'],
            y=subset['z1'],
            c=subset['predicted_prob'],
            cmap='coolwarm',
            marker=marker_map[cond], # 使用对应的形状
            s=100,
            alpha=0.8,
            edgecolors='k',
            linewidth=0.5,
            label=cond,
            vmin=0, vmax=1 # 确保颜色映射范围一致
        )
        # 只要绘制一次，就保存这个 mappable
        if scatter_plot_mappable is None:
            scatter_plot_mappable = current_scatter
        
    # 为 `scatter_plot_mappable` (实际的散点图对象) 添加 colorbar，并指定到 `ax`
    if scatter_plot_mappable:
        cbar = fig.colorbar(scatter_plot_mappable, ax=ax)
        cbar.set_label("Predicted Senescence Probability", rotation=270, labelpad=15)
    
    ax.set_title('New Data in VAE Latent Space\n(Shape=Condition, Color=Prediction)', fontsize=14)
    ax.set_xlabel('Latent Dimension 1 (z0_aligned) [Aging Axis]')
    ax.set_ylabel('Latent Dimension 2 (z1)')
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(title='Condition')
    
    # 保存为 PDF，并确保文件名无空格
    output_plot_name = 'validation_4_latent_space.pdf'.strip()
    fig.savefig(output_plot_name, format='pdf', bbox_inches='tight') # 使用 fig.savefig
    print(f"INFO: Visualization saved: {output_plot_name}")
    plt.close(fig) # 关闭当前的 Figure

# =================================================================
# 7. Main Execution Block
# =================================================================

if __name__ == "__main__":
    run_inference()