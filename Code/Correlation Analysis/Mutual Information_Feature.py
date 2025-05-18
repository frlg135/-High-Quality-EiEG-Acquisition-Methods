import pandas as pd
import os
import numpy as np
from sklearn.feature_selection import mutual_info_regression  # 关键修改1：导入MI计算器

def clean_data(df):
    """数据清洗保持不变"""
    df = df.apply(pd.to_numeric, errors='coerce')
    df.dropna(how='any', inplace=True)
    return df

def calculate_mi(sheep_num, comp_type, input_folder):
    """修改函数名并重构为MI计算"""
    try:
        # 数据读取部分保持不变
        df_a = pd.read_excel(os.path.join(input_folder, f"{sheep_num}A.xlsx"), 
                           header=None, engine='openpyxl')
        df_comp = pd.read_excel(os.path.join(input_folder, f"{sheep_num}{comp_type}.xlsx"),
                              header=None, engine='openpyxl')
        
        df_a = clean_data(df_a)
        df_comp = clean_data(df_comp)
        
        min_length = min(len(df_a), len(df_comp))
        df_a = df_a.iloc[:min_length]
        df_comp = df_comp.iloc[:min_length]

        # 修改结果结构（MI无P值）
        results = pd.DataFrame(columns=['Feature', 'Mutual_Information'])
        
        # 关键修改2：计算每个特征的MI
        for col in range(36):
            try:
                # 将数据转换为二维数组格式
                X = df_a[col].values.reshape(-1, 1)
                y = df_comp[col].values
                mi = mutual_info_regression(X, y, random_state=42)[0]  # 返回数组取第一个值
                results.loc[col] = [f'Feature_{col+1}', mi]
            except Exception as e:
                print(f"Error in {sheep_num}{comp_type} feature {col}: {str(e)}")
                results.loc[col] = [f'Feature_{col+1}', np.nan]

        # 关键修改3：整体MI计算（展平后视为单变量）
        try:
            X_full = df_a.values.flatten().reshape(-1, 1)
            y_full = df_comp.values.flatten()
            overall_mi = mutual_info_regression(X_full, y_full, random_state=42)[0]
        except Exception as e:
            print(f"Overall error in {sheep_num}{comp_type}: {str(e)}")
            overall_mi = np.nan
            
        results.loc[36] = ['Overall', overall_mi]
        return results
    
    except Exception as e:
        print(f"File read error: {str(e)}")
        return pd.DataFrame()

def main():
    input_folder = "Data\Correlation\Feature_data_excel"
    output_folder = "Data\Correlation"
    
    # 修改输出路径
    mi_folder = os.path.join(output_folder, "MI_Results")  # 关键修改4
    os.makedirs(mi_folder, exist_ok=True)

    for sheep in [1, 2]:
        for comp in ['B', 'C']:
            print(f"Processing Sheep{sheep}_A_vs_{comp}...")
            result = calculate_mi(sheep, comp, input_folder)
            if not result.empty:
                # 关键修改5：更新文件名
                output_path = os.path.join(mi_folder, f"Sheep{sheep}_A_vs_{comp}_MI.xlsx")
                result.to_excel(output_path, index=False)
                print(f"Saved: {output_path}")

if __name__ == "__main__":
    main()