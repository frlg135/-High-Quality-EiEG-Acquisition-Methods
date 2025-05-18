import pandas as pd
import numpy as np
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw

def load_data(file_path):
    """增强数据加载与清洗"""
    try:
        # 强制转换数值类型，处理科学计数法
        df = pd.read_excel(file_path, header=None, engine='openpyxl', converters={i: lambda x: pd.to_numeric(x, errors='coerce') for i in range(36)})
        
        # 清除无效列（空值>50%的列）
        df = df.loc[:, df.isnull().mean() < 0.5]
        # 清除无效行
        df = df.dropna(how='any').reset_index(drop=True)
        
        # 维度验证
        if df.shape[0] < 2 or df.shape[1] < 1:
            raise ValueError("数据不足")
        return df.astype(np.float64)
    
    except Exception as e:
        print(f"文件加载失败: {file_path} - {str(e)}")
        return None

def calculate_dtw(file1, file2, output_path):
    df1 = load_data(file1)
    df2 = load_data(file2)
    
    if df1 is None or df2 is None:
        return

    # 列对齐（取最小列数）
    common_cols = min(df1.shape[1], df2.shape[1])
    results = []
    
    for col in range(common_cols):
        try:
            # 增强维度转换 (关键修改点)
            s1 = df1.iloc[:, col].values.astype(float).flatten()  # 强制展平
            s2 = df2.iloc[:, col].values.astype(float).reshape(-1)  # 显式一维化
            
            # 维度断言检查
            assert s1.ndim == 1 and s2.ndim == 1, f"维度错误 s1:{s1.shape}, s2:{s2.shape}"
            
            distance, _ = fastdtw(s1, s2, dist=euclidean)
            results.append({'Feature': f'Col_{col+1}', 'DTW': distance})
        except Exception as e:
            print(f"列 {col+1} 错误详情: {str(e)}")
    
    # 保存结果
    pd.DataFrame(results).to_excel(output_path, index=False)
    print(f"结果已保存: {output_path}")

# 使用示例
calculate_dtw(
    "Data/Correlation/Feature_data_excel/1A.xlsx",
    "Data/Correlation/Feature_data_excel/1B.xlsx",
    "Data/Correlation/DTW_Results/Sheep1_A_vs_B_DTW.xlsx"
)