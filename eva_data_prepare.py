import pandas as pd
import numpy as np
import os
DATA_FOLDER = 'data'
def create_unified_data_from_excel(output_path: str = f'new_p6_data.csv'):
    """
    读取 AFM.xlsx 中 Hypoxia 和 Normoxia 两个子表，合并并标准化列名，
    以生成符合模型输入格式的 CSV 文件。
    """
    
    # =========================================================
    # ⚠️ 必须修改：请替换为您 Excel 文件中 6 种属性的精确列名
    # =========================================================
    # 示例 (请替换为您的实际列名):
    FEATURE_NAMES = ['Adhesion (pN)', 'Elastic modulus（kPa）', 'height', 'roughness', 'length', 'width']
    # =========================================================
    
    csv_filename = f"AFM.xlsx"
    file_path = os.path.join(DATA_FOLDER, csv_filename)
    # 定义 Excel 表格中与模型输入匹配的关键列名
    COLUMN_MAPPING = {
        'Num.': 'cell_id',
        '周期': 'cycle',
        'group': 'Condition'
    }
    
    all_required_cols = list(COLUMN_MAPPING.keys()) + FEATURE_NAMES
    
    try:
        print(f"正在读取文件: {file_path}")
        
        # 1. 读取两个工作表
        df_hypoxia = pd.read_excel(file_path, sheet_name='Hypoxia')
        df_normoxia = pd.read_excel(file_path, sheet_name='Normoxia')
        
    except FileNotFoundError:
        print(f"错误：文件 '{file_path}' 未找到。请确保文件路径和文件名正确。")
        return
    except ValueError as e:
        print(f"错误：读取工作表失败。请检查 Excel 文件中是否有名为 'Hypoxia' 和 'Normoxia' 的子表。详细错误: {e}")
        return
    except Exception as e:
        print(f"读取 Excel 文件时发生错误: {e}")
        return

    # 2. 统一处理两个数据框
    def process_sheet(df: pd.DataFrame, condition_name: str) -> pd.DataFrame:
        df_processed = df.copy()
        
        # 确保所有必需列都存在
        missing_cols = [col for col in all_required_cols if col not in df_processed.columns]
        if missing_cols:
            raise KeyError(f"工作表 '{condition_name}' 缺少必需的列: {missing_cols}。请检查表头是否正确。")
            
        # 标准化列名
        df_processed = df_processed.rename(columns=COLUMN_MAPPING)
        
        # 确保 'Condition' 列被正确设置
        # 如果原始数据中 '组别' 列已经包含 'Hypoxia'/'Normoxia'，则无需修改
        if 'Condition' not in df_processed.columns or df_processed['Condition'].nunique() == 0:
             df_processed['Condition'] = condition_name
             
        # 确保 'cycle' 是字符串
        df_processed['cycle'] = df_processed['cycle'].astype(str)
        
        # 确保 cell_id 是整数
        df_processed['cell_id'] = pd.to_numeric(df_processed['cell_id'], errors='coerce').astype('Int64')

        # 保持模型输入特征的顺序
        final_cols = ['cell_id', 'cycle', 'Condition'] + FEATURE_NAMES
        return df_processed[final_cols]


    try:
        df_hypoxia_processed = process_sheet(df_hypoxia, 'Hypoxia')
        df_normoxia_processed = process_sheet(df_normoxia, 'Normoxia')
    except KeyError as e:
        print(f"列名检查失败: {e}")
        return

    # 3. 最终合并
    df_final = pd.concat([df_hypoxia_processed, df_normoxia_processed], ignore_index=True)

    # 4. 保存为 CSV
    df_final = df_final.sort_values(by=['cell_id', 'Condition']).reset_index(drop=True)
    df_final.to_csv(os.path.join(DATA_FOLDER, output_path), index=False)
    
    print("\n--- 预处理完成 ---")
    print(f"数据合并成功。总行数: {len(df_final)}")
    print(f"已成功生成 '{output_path}' 文件。")
    print("\n--- 新数据预览 (前10行) ---")
    print(df_final.head(10).to_markdown(index=False))


# --- 运行预处理函数 (请确保 AFM.xlsx 文件在同一目录下) ---
create_unified_data_from_excel( output_path=f'new_p6_data.csv')