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
# 1. Configuration and Constants (EN)
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
# 3. Data Transformation Function (CRITICAL FIX HERE)
# =================================================================

def transform_raw_data_to_model_input(df_raw: pd.DataFrame, feature_names: list) -> pd.DataFrame:
    """
    Groups by cell_id, Condition, and cycle to calculate statistics 
    for each unique cell/condition group (FIXED GROUPING KEY).
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


    # Define the CORRECT grouping keys
    GROUPING_KEYS = ['cell_id', 'Condition', 'cycle']
    
    # Drop NaNs from the essential grouping columns and features
    df_temp.dropna(subset=GROUPING_KEYS + feature_names, inplace=True) 
    
    print(f"DEBUG: df_temp shape after dropping NaNs in grouping keys/features: {df_temp.shape}")
    print(f"DEBUG: Unique 'Condition' values in df_temp after cleaning: {df_temp['Condition'].unique().tolist()}")

    if df_temp.empty:
        raise ValueError("ERROR: Dataframe is empty after cleaning essential columns.")

    df_temp['cell_id'] = df_temp['cell_id'].astype(int)
    
    # A. Aggregate Feature Columns (numeric stats) - NOW GROUPING BY CELL_ID AND CONDITION
    agg_funcs = {name: ['mean', 'median', 'std'] for name in feature_names}
    # **CRITICAL FIX: Group by the composite key**
    df_vae_input = df_temp.groupby(GROUPING_KEYS).agg(agg_funcs).reset_index() 
    
    # Correct column flattening after grouping by multiple keys
    # The first N columns are the GROUPING_KEYS, the rest are MultiIndex
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
        print(f"INFO: Scaler '{scaler_path}' loaded successfully.")
    except Exception as e:
        print(f"WARNING: Failed to load scaler. Detail: {e}. Proceeding WITHOUT standardization.")

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
# 5. Main Inference Function
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

    # 4. Model Inference & Latent Space Standardization (KEY FIX)
    print("\n--- 4. Model Inference ---")
    X_tensor = torch.tensor(X_new, dtype=torch.float32)
    with torch.no_grad():
        _, z_mu_new, _, _ = vae(X_tensor) 

    z_mu_new_np = z_mu_new.cpu().numpy()
    
    # --- KEY FIX: Standardize the Latent Space for Regressor ---
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
    
    df_results['z0'] = z_mu_new_scaled_np[:, 0] * flip_factor 
    
    # 6. Validation and Plotting
    plot_and_validate(df_results)
    print("--- INFERENCE COMPLETE ---")


# =================================================================
# 6. Visualization and Validation Function (EN)
# =================================================================

def plot_and_validate(df_results: pd.DataFrame):
    """Generates a box plot and performs validation check (EN)."""
        
    mean_scores = df_results.groupby('Condition')['predicted_prob'].mean().sort_values(ascending=False)
    
    print("\n--- 5. Results Validation ---")
    print("Mean Predicted Senescence Score:")
    print(mean_scores.to_string()) 
    
    validation_success = False
    available_conditions = df_results['Condition'].unique().tolist()

    try:
        if 'Hypoxia' in available_conditions and 'Normoxia' in available_conditions:
            # Expected result: Hypoxia < Normoxia (Based on common senescence models)
            if mean_scores['Hypoxia'] < mean_scores['Normoxia']:
                print("\nSUCCESS: Hypoxia mean probability is lower than Normoxia, as expected.")
                validation_success = True
            else:
                print("\nFAILURE: Hypoxia mean probability is NOT lower than Normoxia.")
        else:
            print(f"\nWARNING: Could not perform validation check. Found conditions: {available_conditions}. Need both 'Normoxia' and 'Hypoxia'.")
    except Exception:
        print("\nWARNING: An error occurred during validation check.")
    
    plt.figure(figsize=(6, 5))
    
    plot_order = [cond for cond in ['Normoxia', 'Hypoxia'] if cond in available_conditions]
    
    if not plot_order:
         print("\nFATAL ERROR: No valid conditions (Hypoxia/Normoxia) found for plotting.")
         plt.close()
         return

    palette_map = {'Normoxia': 'r', 'Hypoxia': 'b'}
    
    sns.boxplot(
        x='Condition', 
        y='predicted_prob', 
        data=df_results, 
        order=plot_order, 
        palette={k: palette_map[k] for k in plot_order}
    )
    plt.title('Predicted Senescence Probability by Oxygen Condition')
    plt.xlabel('Condition')
    plt.ylabel('Senescence Probability')
    
    for i, condition in enumerate(plot_order):
        score = mean_scores.get(condition, np.nan)
        if not np.isnan(score):
            # Adjust text position slightly above the max point
            max_prob = df_results[df_results['Condition'] == condition]['predicted_prob'].max()
            text_y_pos = max_prob + (df_results['predicted_prob'].max() - df_results['predicted_prob'].min()) * 0.05
            plt.text(i, text_y_pos, f'Mean: {score:.2f}', 
                     horizontalalignment='center', color='black', fontsize=10)

    output_plot_name = 'validation_boxplot_hypoxia.pdf  '
    plt.savefig(output_plot_name, dpi=300)
    print(f"\nINFO: Visualization saved to: {output_plot_name}")
    plt.close()

# =================================================================
# 7. Main Execution Block
# =================================================================

if __name__ == "__main__":
    run_inference()