import pandas as pd
from scipy.stats import pearsonr
import os
import numpy as np

def clean_data(df):
    """将数据强制转换为数值类型并处理空值"""
    # 转换所有列为数值类型
    df = df.apply(pd.to_numeric, errors='coerce')
    # 删除包含空值的行（保持样本对齐）
    df.dropna(how='any', inplace=True)
    return df

def calculate_correlations(sheep_num, comp_type, input_folder):
    """计算介入式和非介入式数据的相关系数"""
    try:
        # 读取介入式数据（假设无表头）
        df_a = pd.read_excel(os.path.join(input_folder, f"{sheep_num}A.xlsx"), 
                            header=None,
                            engine='openpyxl')
        # 读取非介入式数据（B或C）
        df_comp = pd.read_excel(os.path.join(input_folder, f"{sheep_num}{comp_type}.xlsx"),
                               header=None,
                               engine='openpyxl')
        
        # 数据清洗
        df_a = clean_data(df_a)
        df_comp = clean_data(df_comp)
        
        # 对齐数据长度（取最小值）
        min_length = min(len(df_a), len(df_comp))
        df_a = df_a.iloc[:min_length]
        df_comp = df_comp.iloc[:min_length]

        # 初始化结果存储
        results = pd.DataFrame(columns=['Feature', 'Correlation', 'P-Value'])
        
        # 计算每个特征的相关系数
        for col in range(36):
            try:
                corr, p_value = pearsonr(df_a[col].values, df_comp[col].values)
                results.loc[col] = [f'Feature_{col+1}', corr, p_value]
            except Exception as e:
                print(f"Error in {sheep_num}{comp_type} feature {col}: {str(e)}")
                results.loc[col] = [f'Feature_{col+1}', np.nan, np.nan]

        # 计算整体相关系数
        try:
            overall_corr, overall_p = pearsonr(df_a.values.flatten(), df_comp.values.flatten())
        except Exception as e:
            print(f"Overall error in {sheep_num}{comp_type}: {str(e)}")
            overall_corr, overall_p = (np.nan, np.nan)
            
        results.loc[36] = ['Overall', overall_corr, overall_p]
        return results
    
    except Exception as e:
        print(f"File read error: {str(e)}")
        return pd.DataFrame()

def main():
    input_folder = "Data\Correlation\Feature_data_excel"  # 替换为实际路径
    output_folder = "Data\Correlation"  # 替换为实际路径

    os.makedirs(output_folder, exist_ok=True)

    # 处理所有组合
    for sheep in [1, 2]:
        for comp in ['B', 'C']:
            print(f"Processing Sheep{sheep}_A_vs_{comp}...")
            result = calculate_correlations(sheep, comp, input_folder)
            if not result.empty:
                output_path = os.path.join(output_folder, f"Sheep{sheep}_A_vs_{comp}.xlsx")
                result.to_excel(output_path, index=False)
                print(f"Saved: {output_path}")
            else:
                print(f"Failed to process Sheep{sheep}_A_vs_{comp}")

if __name__ == "__main__":
    main()