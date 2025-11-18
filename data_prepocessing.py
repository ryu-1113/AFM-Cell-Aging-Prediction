import pandas as pd
import numpy as np
import os
data_folder = 'data'

def preprocess_raw_data_with_id_corrected():
    """
    加载 8000 条测量的原始数据 (包含 'cell_id')，并将其聚合成 400 条细胞数据。

    Args:
        raw_csv_path (str): 原始数据文件路径 (8000 行)。
        output_csv_path (str): 处理后聚合数据的文件路径 (400 行)。
    """
    
    # --- 1. 数据加载与校验 ---
    csv_filename = f"all.csv"
    try:
        df = pd.read_csv(os.path.join(data_folder, csv_filename))
    except FileNotFoundError:
        print(f"File not found: {csv_filename}. ")

    print(f"Loaded raw data with {len(df)} rows.")

    if 'cell_id' not in df.columns or 'cycle' not in df.columns:
        raise ValueError("Error: 'cell_id' or 'cycle' column not found in the raw data.")

    # 定义 6 个形态/力学特征
    feature_cols = ['height', 'roughness', 'length', 'wideth', 'adhesion', 'elastic_modulus']
    
    # 校验特征列
    missing_cols = [col for col in feature_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Error: Missing feature columns in data: {missing_cols}")

    # --- 2. 构建 Aggregation 字典并执行 ---
    
    agg_funcs = ['mean', 'std', 'median']
    agg_dict = {}
    
    # 添加 6 个特征列的 3 个统计量
    for feature in feature_cols:
        agg_dict[feature] = agg_funcs
        
    # 添加 'cycle' 列，取分组后的第一个值
    agg_dict['cycle'] = 'first'
    
    print(f"Aggregating {len(feature_cols)} features by 'cell_id' (Total {len(df['cell_id'].unique())} unique cells)...")
    
    # 执行 Aggregation
    # 使用 as_index=False 确保 'cell_id' 立即成为普通列，而不是索引
    df_agg = df.groupby('cell_id', as_index=False).agg(agg_dict)
    
    # --- 3. 清理 DataFrame 列名 (重点修正区) ---
    
    # 聚合后的列名现在是 (feature, stat) 或 (cycle, first) 的 MultiIndex，但 'cell_id' 是单字符串
    
    
    new_columns = ['cell_id'] # 强制第一个列名为 'cell_id'
    for col in df_agg.columns[1:]:  
        if isinstance(col, tuple):
        # 这是一个 MultiIndex 列 (例如: ('height', 'mean'), ('cycle', 'first'))
        
        # 1. 检查是否为 'cycle' 列的聚合结果
            if col[1] == 'first':
                # 将 ('cycle', 'first') 命名为 'cycle'
                new_columns.append(col[0])
            else:
                # 将 ('feature', 'stat') 命名为 'feature_stat'
                new_columns.append(f"{col[0]}_{col[1]}")
        else:
        # 这是一个单字符串列 (例如: 'cell_id'，因为它来自 reset_index())
        # 保持原样，防止被错误命名为 cell_id_
            new_columns.append(col)

    # 应用新的列名
    if len(new_columns) == len(df_agg.columns):
        df_agg.columns = new_columns
        print(f"Successfully renamed all {len(new_columns)} columns.")
    else:
    # 如果仍然出现长度不匹配，则立即报错，并打印详细信息以供调试
        raise ValueError(
            f"Column renaming failed: Expected {len(df_agg.columns)} columns, "
            f"but generated {len(new_columns)} new names."
            f"Existing columns: {df_agg.columns.tolist()}"
            f"Generated names: {new_columns}"
        )
        
    df_agg.columns = new_columns
    print(f"Successfully renamed columns. Final shape: {df_agg.shape}") 
    
    # --- 4. 保存结果 ---
    
    # --- Final Steps ---
    output_csv_filename = f"agg.csv"
    output_csv_path = os.path.join(data_folder, output_csv_filename)
    # Re-order columns for clarity
    agg_feature_cols = [col for col in df_agg.columns if col not in ['cell_id', 'cycle']]
    final_cols = ['cell_id', 'cycle'] + agg_feature_cols
    df_agg = df_agg[final_cols]
    
    print(f"Aggregation complete. Final shape: {df_agg.shape}") 
    
    df_agg.to_csv(output_csv_path, index=False)
    print(f"Aggregated cell data saved to '{output_csv_path}'.")
    
    return df_agg

# --- 如何运行 ---
if __name__ == '__main__':
    # 注意：请将 'raw_data.csv' 替换为您实际的 8000 行数据文件。
    try:
        df_cell_data = preprocess_raw_data_with_id_corrected(
            raw_csv_path='raw_data.csv',
            output_csv_path='cell_data.csv'
        )
        print("\nPreprocessing finished successfully. Sample data:")
        print(df_cell_data.head())
        print(f"\nFinal columns ({len(df_cell_data.columns)} total):")
        print(df_cell_data.columns.tolist())
    except Exception as e:
        print(f"\nFATAL ERROR during preprocessing: {e}")

# --- 如何运行 ---
if __name__ == '__main__':
    
    df_cell_data = preprocess_raw_data_with_id_corrected(
    )
    
    print("\nPreprocessing finished. Sample of the output 'cell_data.csv':")
    print(df_cell_data.head())
    
    print(f"\nOutput shape: {df_cell_data.shape}") # 应为 (400, 20)
    print("\nOutput columns:")
    print(df_cell_data.columns.tolist())