import pandas as pd
from scipy.stats import spearmanr  # 修改1：替换为spearmanr
import os
import numpy as np

def clean_data(df):
    """将数据强制转换为数值类型并处理空值"""
    df = df.apply(pd.to_numeric, errors='coerce')
    df.dropna(how='any', inplace=True)
    return df

def calculate_correlations(sheep_num, comp_type, input_folder):
    """计算介入式和非介入式数据的相关系数"""
    try:
        # 读取数据部分保持不变
        df_a = pd.read_excel(os.path.join(input_folder, f"{sheep_num}A.xlsx"), 
                            header=None,
                            engine='openpyxl')
        df_comp = pd.read_excel(os.path.join(input_folder, f"{sheep_num}{comp_type}.xlsx"),
                               header=None,
                               engine='openpyxl')
        
        df_a = clean_data(df_a)
        df_comp = clean_data(df_comp)
        
        min_length = min(len(df_a), len(df_comp))
        df_a = df_a.iloc[:min_length]
        df_comp = df_comp.iloc[:min_length]

        results = pd.DataFrame(columns=['Feature', 'Correlation', 'P-Value'])
        
        # 修改2：替换为spearmanr
        for col in range(36):
            try:
                corr, p_value = spearmanr(df_a[col].values, df_comp[col].values)  # 关键修改
                results.loc[col] = [f'Feature_{col+1}', corr, p_value]
            except Exception as e:
                print(f"Error in {sheep_num}{comp_type} feature {col}: {str(e)}")
                results.loc[col] = [f'Feature_{col+1}', np.nan, np.nan]

        # 修改3：整体计算使用spearmanr
        try:
            overall_corr, overall_p = spearmanr(df_a.values.flatten(), df_comp.values.flatten())
        except Exception as e:
            print(f"Overall error in {sheep_num}{comp_type}: {str(e)}")
            overall_corr, overall_p = (np.nan, np.nan)
            
        results.loc[36] = ['Overall', overall_corr, overall_p]
        return results
    
    except Exception as e:
        print(f"File read error: {str(e)}")
        return pd.DataFrame()

def main():
    input_folder = "Data\Correlation\Feature_data_excel"
    output_folder = "Data\Correlation"
    
    # 修改4：添加子目录避免覆盖旧结果
    spearman_folder = os.path.join(output_folder, "Spearman_Results")
    os.makedirs(spearman_folder, exist_ok=True)

    for sheep in [1, 2]:
        for comp in ['B', 'C']:
            print(f"Processing Sheep{sheep}_A_vs_{comp}...")
            result = calculate_correlations(sheep, comp, input_folder)
            if not result.empty:
                # 修改5：修改输出文件名
                output_path = os.path.join(spearman_folder, f"Sheep{sheep}_A_vs_{comp}_spearman.xlsx")
                result.to_excel(output_path, index=False)
                print(f"Saved: {output_path}")

if __name__ == "__main__":
    main()