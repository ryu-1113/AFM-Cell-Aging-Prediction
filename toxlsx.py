import pandas as pd
import os

def xlsx_sheets_to_csvs(xlsx_path, output_folder=None):
    # 读取所有子表
    sheets = pd.read_excel(xlsx_path, sheet_name=None)
    # 默认输出路径为xlsx所在目录
    if output_folder is None:
        output_folder = os.path.dirname(xlsx_path) or '.'
    
    # 获取不带扩展名的文件名
    base_name = os.path.splitext(os.path.basename(xlsx_path))[0]
    
    for sheet_name, df in sheets.items():
        # 构造输出文件路径
        csv_filename = f"{base_name}_{sheet_name}.csv"
        csv_path = os.path.join(output_folder, csv_filename)
        # 保存为CSV
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"✅ 已转换子表: {sheet_name} → {csv_filename}")

# 示例
xlsx_sheets_to_csvs('C:/code/data/UCSC.xlsx')