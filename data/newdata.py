import pandas as pd
import numpy as np
import os
def create_unified_data_from_excel(file_path: str = f'AFM.xlsx', output_path: str = f'new_p6_data.csv'):
    """
    Reads Hypoxia and Normoxia sheets from AFM.xlsx, merges them, and saves
    the data into a standardized CSV file for model inference.
    
    NOTE: The column names in FEATURE_NAMES MUST exactly match the headers in your Excel file.
    """
    data_folder = 'data'
    # =========================================================
    # ⚠️ CRITICAL: Replace with the EXACT column names of your 6 properties
    # 示例 (Please replace with your actual column names):
    FEATURE_NAMES = ['Adhesion (pN)', 'Elastic modulus（kPa）', 'height', 'roughness', 'length', 'width']
    # =========================================================
    
    
    # Mapping for standardized column names (English names for internal use)
    COLUMN_MAPPING = {
        'Num.': 'cell_id',
        'cycle': 'cycle',
        'group': 'Condition'
    }
    
    all_required_cols = list(COLUMN_MAPPING.keys()) + FEATURE_NAMES
    
    try:
        print(f"INFO: Reading file: {file_path}")
        
        # 1. Read both sheets
        df_hypoxia = pd.read_excel(os.path.join(data_folder, file_path), sheet_name='Hypoxia')
        df_normoxia = pd.read_excel(os.path.join(data_folder, file_path), sheet_name='Normoxia')
        
    except FileNotFoundError:
        print(f"ERROR: File '{file_path}' not found. Check file path.")
        return
    except ValueError as e:
        print(f"ERROR: Failed to read sheets. Check if 'Hypoxia' and 'Normoxia' sheets exist.")
        # print(f"DETAIL: {e}") # Removed detailed error to avoid complex characters
        return
    except Exception as e:
        print(f"ERROR: An unknown error occurred during Excel reading.")
        return

    # 2. Process each DataFrame
    def process_sheet(df: pd.DataFrame, condition_name: str) -> pd.DataFrame:
        df_processed = df.copy()
        
        # Check for required columns
        missing_cols = [col for col in all_required_cols if col not in df_processed.columns]
        if missing_cols:
            raise KeyError(f"Sheet '{condition_name}' is missing columns: {missing_cols}")
            
        # Standardize column names
        df_processed = df_processed.rename(columns=COLUMN_MAPPING)
        
        # Ensure 'Condition' column is set
        if 'Condition' not in df_processed.columns or df_processed['Condition'].nunique() == 0:
             df_processed['Condition'] = condition_name
             
        # Data type cleaning
        df_processed['cycle'] = df_processed['cycle'].astype(str)
        df_processed['cell_id'] = pd.to_numeric(df_processed['cell_id'], errors='coerce').astype('Int64')

        # Select and reorder columns
        final_cols = ['cell_id', 'cycle', 'Condition'] + FEATURE_NAMES
        return df_processed[final_cols]


    try:
        df_hypoxia_processed = process_sheet(df_hypoxia, 'Hypoxia')
        df_normoxia_processed = process_sheet(df_normoxia, 'Normoxia')
    except KeyError as e:
        print(f"ERROR: Column name check failed: {e}")
        return

    # 3. Final Concatenation
    df_final = pd.concat([df_hypoxia_processed, df_normoxia_processed], ignore_index=True)

    # 4. Save to CSV
    df_final = df_final.sort_values(by=['cell_id', 'Condition']).reset_index(drop=True)
    
    df_final = df_final.rename(columns={
        'height': 'Height',
        'roughness':'Roughness',
        'Adhesion (pN)':'Adhesion',
        'Elastic modulus（kPa）': 'Elastic_modulus'  
    })   

    df_final.columns = df_final.columns.str.strip()  #去除列名中的空格
    df_final.columns = df_final.columns.str.replace(' ','_')  #使用_取代空格
    df_final.columns = df_final.columns.str.lower() #全小写的 

    df_final.to_csv(output_path, index=False)
    
    print("\n--- PREPROCESSING COMPLETE ---")
    print(f"INFO: Data merged successfully. Total rows: {len(df_final)}")
    print(f"INFO: Output saved to '{output_path}'")
    print("\n--- DATA PREVIEW (First 10 Rows) ---")
    # Using to_markdown for clean, non-Chinese table output
    print(df_final.head(10).to_markdown(index=False))


# --- 运行预处理函数 (Run the preprocessing function) ---
create_unified_data_from_excel(file_path='AFM.xlsx', output_path='new_p6_data.csv')