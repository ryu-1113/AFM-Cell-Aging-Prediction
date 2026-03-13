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
from typing import Tuple, List, Dict

# =================================================================
# 1. Configuration and Constants
# =================================================================

RAW_DATA_PATH = os.path.join('data','new_p6_data.csv' ) 
META_FILE_PATH = 'inference_meta.json'
VAE_MODEL_PATH = 'best_vae_model.pth'
REGRESSOR_MODEL_PATH = 'best_regressor_model.pth'
SCALER_PATH = 'scaler.pkl' 

OUTPUT_DIR = 'result' 
os.makedirs(OUTPUT_DIR, exist_ok=True) 

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
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

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
# 3. Data Transformation Function (MODIFIED: No Aggregation)
# =================================================================

def transform_raw_data_to_model_input(df_raw: pd.DataFrame, feature_names: list) -> pd.DataFrame:
    """
    Convert raw rows directly to model input format WITHOUT aggregation.
    Each row in the CSV becomes one sample.
    mean = raw_value, median = raw_value, std = 0.0
    """
    df_temp = df_raw.copy()
    
    print(f"DEBUG: Inside transform function - initial df_temp shape: {df_temp.shape}")

    # 1. Fill metadata (Condition, cycle, cell_id) downwards
    # This ensures every row has a label, even if the CSV merged cells visually
    cols_to_fill = ['Condition', 'cycle', 'cell_id']
    for col in cols_to_fill:
        if col in df_temp.columns:
            df_temp[col] = df_temp[col].ffill()
            
    print("INFO: Forward-filled metadata columns (Condition, cycle, cell_id) to ensure every row is used.")

    # 2. Drop rows where essential feature data is missing
    df_temp.dropna(subset=feature_names, inplace=True)
    
    # 3. Create the 18 model features directly from the 6 raw features
    # Since we are treating each row as a sample, there is no "group".
    # Mean = Value, Median = Value, Std = 0
    
    for name in feature_names:
        df_temp[f'{name}_mean'] = df_temp[name]
        df_temp[f'{name}_median'] = df_temp[name]
        df_temp[f'{name}_std'] = 0.0 # Single measurement has zero variance
    
    # Final check
    # Ensure we have all required input columns
    df_final = df_temp.copy()
    
    # Keep only relevant columns
    keep_cols = ['Condition', 'cycle'] + MODEL_INPUT_FEATURES
    # If cell_id exists, keep it for reference, though not used for grouping anymore
    if 'cell_id' in df_final.columns:
        keep_cols.insert(0, 'cell_id')
        
    df_final = df_final[keep_cols]

    print(f"DEBUG: Final transformed shape (Rows x Features): {df_final.shape}")
    print(f"INFO: Each of the {len(df_final)} rows will be predicted independently.")
    
    return df_final

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
    
    return vae, regressor, scaler 

# =================================================================
# 5. Main Inference Function
# =================================================================

def run_inference():
    """Executes the full inference pipeline."""
    
    # 1. Load resources
    vae, regressor, scaler = load_models_and_metadata(
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

    # Simplify cycle column renaming
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

    # **CHANGED**: Uses the non-aggregating transformation
    df_vae_input = transform_raw_data_to_model_input(df_raw, FEATURE_NAMES)
    print(f"INFO: Feature transformation complete. Generated {len(df_vae_input)} entries (No aggregation).")
    
    X_new = df_vae_input[MODEL_INPUT_FEATURES].values
    
    # 3. Standardization (Raw Features)
    if scaler is not None:
        X_new = scaler.transform(X_new)
        print("INFO: Raw features standardized using loaded scaler.")
    else:
        print("WARNING: Raw feature standardization skipped.")

    # 4. Model Inference & Latent Space Standardization
    print("\n--- 4. Model Inference ---")
    X_tensor = torch.tensor(X_new, dtype=torch.float32)
    with torch.no_grad():
        _, z_mu_new, _, _ = vae(X_tensor) 

    z_mu_new_np = z_mu_new.cpu().numpy()
    
    # --- ORIGINAL KEY FIX: Standardize the Latent Space for Regressor ---
    z_scaler = StandardScaler() 
    z_mu_new_scaled_np = z_scaler.fit_transform(z_mu_new_np)
    
    z_mu_new_scaled = torch.tensor(z_mu_new_scaled_np, dtype=torch.float32)
    print(f"DEBUG: VAE latent mu (z_mu_new) min/max BEFORE Z-scaling: {z_mu_new_np.min():.4f} / {z_mu_new_np.max():.4f}")
    print(f"INFO: Applied **Z-space standardization** (Mean=0, Std=1) to prevent Regressor saturation.")
    print(f"DEBUG: VAE latent mu (z_mu_new) min/max AFTER Z-scaling: {z_mu_new_scaled.min().item():.4f} / {z_mu_new_scaled.max().item():.4f}")
    
    with torch.no_grad():
        predicted_prob_tensor = regressor(z_mu_new_scaled) 
    
    # 5. Result Integration
    predicted_prob_np = predicted_prob_tensor.cpu().numpy().flatten()
    print(f"DEBUG: Predicted probabilities (raw tensor) min/max AFTER Z-scaling: {predicted_prob_np.min():.4f} / {predicted_prob_np.max():.4f}")
    
    df_results = df_vae_input.copy()
    df_results['predicted_prob'] = predicted_prob_np
    
    # --- AXIS MAPPING LOGIC ---
    # z0 = -z1 (Aging Axis)
    df_results['z0'] = -z_mu_new_np[:, 1] 
    # z1 = z0 
    df_results['z1'] = z_mu_new_np[:, 0] 

    # 6. Validation and Plotting
    plot_and_validate(df_results)
    print("--- INFERENCE COMPLETE ---")


# =================================================================
# 6. Visualization and Validation Function
# =================================================================

def plot_and_validate(df_results: pd.DataFrame):
    """Generates multiple plots and performs validation check (EN)."""
    BW_ADJUST_FACTOR = 1    
    mean_scores = df_results.groupby('Condition')['predicted_prob'].mean().sort_values(ascending=False)
    
    print("\n--- 5. Results Validation ---")
    print("Mean Predicted Senescence Score:")
    print(mean_scores.to_string()) 
    print(f"Total Data Points: {len(df_results)}")
    print(f"Points per Condition:\n{df_results['Condition'].value_counts().to_string()}")
    
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
    
    # Plot Setup
    plot_order = [cond for cond in ['Normoxia', 'Hypoxia'] if cond in available_conditions]
    if not plot_order:
         print("\nFATAL ERROR: No valid conditions found.")
         plt.close()
         return

    palette_map = {'Normoxia': 'r', 'Hypoxia': 'b'}
    marker_map = {'Normoxia': 'o', 'Hypoxia': 'X'} 

    # --- Plot 1: Boxplot ---
    plt.figure(figsize=(6, 5))
    sns.boxplot(
        x='Condition', y='predicted_prob', data=df_results, 
        order=plot_order, palette={k: palette_map[k] for k in plot_order}
    )
    plt.title('Predicted Senescence Probability by Oxygen Condition (Boxplot)')
    plt.xlabel('Condition'); plt.ylabel('Senescence Probability')
    plt.ylim(-0.05, 1.05)
    
    for i, condition in enumerate(plot_order):
        score = mean_scores.get(condition, np.nan)
        if not np.isnan(score):
            plt.text(i, 1.03, f'Mean: {score:.3f}', horizontalalignment='center', color='black', fontsize=10)

    plt.savefig(os.path.join(OUTPUT_DIR,'validation_1_boxplot.pdf'), format='pdf', bbox_inches='tight')
    print("INFO: Saved validation_1_boxplot.pdf")
    plt.close()

    # --- Plot 2: Violin Plot ---
    plt.figure(figsize=(6, 5))
    sns.violinplot(
        x='Condition', y='predicted_prob', data=df_results, 
        order=plot_order, bw_adjust=BW_ADJUST_FACTOR, palette={k: palette_map[k] for k in plot_order}, inner='box'
    )
    plt.title('Predicted Senescence Probability by Oxygen Condition (Violin Plot)')
    plt.xlabel('Condition'); plt.ylabel('Senescence Probability')
    plt.ylim(-0.05, 1.05)
    plt.savefig(os.path.join(OUTPUT_DIR,'validation_2_violinplot.pdf'), format='pdf', bbox_inches='tight')
    print("INFO: Saved validation_2_violinplot.pdf")
    plt.close()

    # --- Plot 3: Density Plot ---
    plt.figure(figsize=(7, 5))
    for cond in plot_order:
        subset = df_results[df_results['Condition'] == cond]
        if len(subset) > 1:
            sns.kdeplot(subset['predicted_prob'], label=cond, color=palette_map[cond], bw_adjust=BW_ADJUST_FACTOR, fill=True, alpha=0.4)
    
    plt.title('Predicted Senescence Probability Density')
    plt.xlabel('Senescence Probability'); plt.ylabel('Density')
    plt.xlim(-0.05, 1.05); plt.legend(title='Condition')
    plt.savefig(os.path.join(OUTPUT_DIR,'validation_3_densityplot.pdf'), format='pdf', bbox_inches='tight')
    print("INFO: Saved validation_3_densityplot.pdf")
    plt.close()

    # --- Plot 4: Latent Space Projection ---
    fig, ax = plt.subplots(figsize=(8, 6)) 
    scatter_plot_mappable = None
    
    for cond in plot_order:
        subset = df_results[df_results['Condition'] == cond]
        # Scatter with NO aggregation (all points shown)
        current_scatter = ax.scatter(
            x=subset['z0'], y=subset['z1'], c=subset['predicted_prob'],
            cmap='coolwarm', marker=marker_map[cond], 
            s=60, alpha=0.7, edgecolors='k', linewidth=0.3,
            label=cond, vmin=0, vmax=1 
        )
        if scatter_plot_mappable is None:
            scatter_plot_mappable = current_scatter
        
    if scatter_plot_mappable:
        cbar = fig.colorbar(scatter_plot_mappable, ax=ax)
        cbar.set_label("Predicted Senescence Probability", rotation=270, labelpad=15)
    
    ax.set_title('New Data in VAE Latent Space\n(Shape=Condition, Color=Prediction)', fontsize=14)
    ax.set_xlabel('Latent Dimension 1 (z0) [Aging Axis]')
    ax.set_ylabel('Latent Dimension 2 (z1)')
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(title='Condition')
    
    fig.savefig(os.path.join(OUTPUT_DIR,'validation_4_latent_space.pdf'), format='pdf', bbox_inches='tight') 
    print("INFO: Saved validation_4_latent_space.pdf")
    plt.close(fig) 

# =================================================================
# 7. Main Execution Block
# =================================================================

if __name__ == "__main__":
    run_inference()