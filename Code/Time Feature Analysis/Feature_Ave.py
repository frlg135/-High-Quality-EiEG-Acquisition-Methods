#用来求出所有特征的均值和标准差，用在时域分析和频域分析上
#使用的是相关性用到的数据，因为其被处理的最好，最方便

import pandas as pd
import numpy as np
import os

def calculate_column_stats(file_path):
    """
    计算Excel文件中每列的统计量，包含科学计数法格式和组合格式
    :param file_path: Excel文件路径
    :return: 包含统计结果的DataFrame
    """
    # 读取Excel文件（假设没有标题行）
    df = pd.read_excel(file_path, header=None)
    
    # 数据清洗：强制转换为数值类型
    df = df.apply(pd.to_numeric, errors='coerce')
    
    # 验证数据维度
    if df.shape != (1200, 36):
        print(f"警告: 文件维度为{df.shape}，预期为(1200, 36)")
    
    # 计算统计量
    means = df.mean().values
    stds = df.std(ddof=0).values  # 使用总体标准差
    
    # 格式化字符串
    formatted_means = ["{:.4g}".format(x) for x in means]
    formatted_stds = ["{:.4g}".format(x) for x in stds]
    combined = [f"{m}±{s}" for m, s in zip(formatted_means, formatted_stds)]
    
    # 创建结果DataFrame
    stats = pd.DataFrame({
        '列序号': range(1, 37),
        '均值': formatted_means,
        '标准差': formatted_stds,
        '均值±标准差': combined
    })
    
    return stats

# 使用示例
if __name__ == "__main__":
    input_file = r"Data\Correlation\Feature_data_excel\1A.xlsx"  # 替换为实际路径
    output_file = "1A.xlsx"

    try:
        # 执行计算
        result = calculate_column_stats(input_file)
        
        # 保存结果
        result.to_excel(output_file, index=False)
        print(f"计算完成！结果已保存至: {os.path.abspath(output_file)}")
        print("\n统计结果预览:")
        print(result.iloc[:10])  # 显示前10行结果
        
    except Exception as e:
        print(f"处理失败: {str(e)}")
        print("请检查: 1.文件路径 2.文件格式 3.Excel文件是否损坏")