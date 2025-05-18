import os
import re
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler

plt.rcParams['font.size'] = 20
def load_data(folder):
    """加载36列特征数据"""
    data = {}
    pattern = re.compile(r'^(\d+)([ABC])\.xlsx$', re.IGNORECASE)
    
    # 新增特征名称映射列表
    feature_names = [
        'SampEn', 'MAV', 'VAR', 'RMS', 'LOG', 'WL', 'STD', 'ADSD', 'FD', 'MFL',
        'MTO', 'IEEG', 'SEEG', 'ZCR', 'SSC', 'WAMP', 'AC_X1', 'AC_X2', 'AC_X3',
        'AC_X4', 'AC_X5', 'Pmax', 'Fmax', 'MP', 'TP', 'MNF', 'MDF', 'FSTD',
        'SM1', 'SM2', 'SM3', 'KUR', 'SKW', 'CC1', 'CC2', 'CC3'
    ]  # 共36个特征
    
    for file in os.listdir(folder):
        match = pattern.match(file)
        if not match:
            continue
            
        # 解析文件信息
        sheep_id = f"Sheep{match.group(1)}"
        signal_type = {'A': 'Intervention', 'B': 'Left', 'C': 'Right'}[match.group(2).upper()]
        
        try:
            # 加载数据并验证列数
            df = pd.read_excel(os.path.join(folder, file), header=None)
            if df.shape[1] != 36:
                print(f"文件{file}列数异常 ({df.shape[1]}列)，已跳过")
                continue
                
            # 转换为数值类型并应用新特征名称
            df = df.apply(pd.to_numeric, errors='coerce')
            df.columns = feature_names  # 关键修改点：替换原特征名为新名称
            
            # 存储数据
            key = (sheep_id, signal_type)
            data[key] = df.dropna(how='all')  # 删除全空行
            
        except Exception as e:
            print(f"加载{file}失败: {str(e)}")
    
    return data

def calculate_correlation(src_df, tgt_df):
    """计算标准化后的相关系数矩阵"""
    # 样本对齐
    min_samples = min(len(src_df), len(tgt_df))
    src = src_df.iloc[:min_samples]
    tgt = tgt_df.iloc[:min_samples]
    
    # 标准化处理
    scaler = StandardScaler()
    src_scaled = scaler.fit_transform(src)
    tgt_scaled = scaler.transform(tgt)
    
    # 计算相关系数矩阵
    corr_matrix = np.zeros((36, 36))
    for i in range(36):
        for j in range(36):
            valid = ~np.isnan(src_scaled[:,i]) & ~np.isnan(tgt_scaled[:,j])
            if valid.sum() >= 10:  # 至少10个有效样本
                corr_matrix[i,j], _ = pearsonr(src_scaled[valid,i], tgt_scaled[valid,j])
            else:
                corr_matrix[i,j] = np.nan
                
    return pd.DataFrame(corr_matrix,
                       index=src.columns,
                       columns=tgt.columns)

def save_results(sheep_id, comparison, corr_matrix, output_folder):
    """保存结果到Excel和图片"""
    # 创建输出目录
    os.makedirs(output_folder, exist_ok=True)
    
    # 保存Excel
    excel_path = os.path.join(output_folder, f"{sheep_id}_{comparison}.xlsx")
    corr_matrix.to_excel(excel_path)
    
    # 生成热力图
    plt.figure(figsize=(16, 14))
    features = [
    'SampEn', 'MAV', 'VAR', 'RMS', 'LOG', 'WL', 'STD', 'ADSD', 'FD', 'MFL',
    'MTO', 'IEEG', 'SEEG', 'ZCR', 'SSC', 'WAMP', 'AC_X1', 'AC_X2', 'AC_X3',
    'AC_X4', 'AC_X5', 'Pmax', 'Fmax', 'MP', 'TP', 'MNF', 'MDF', 'FSTD',
    'SM1', 'SM2', 'SM3', 'KUR', 'SKW', 'CC1', 'CC2', 'CC3'
]

    
    sns.heatmap(corr_matrix, 
                cmap='coolwarm',
                center=0,
                annot=False,
                mask=np.isnan(corr_matrix),
                cbar_kws={'label': 'Pearson r'})
    plt.title(f"{sheep_id} {comparison}\nCorrelation Matrix", fontsize=20)
    plt.xlabel('Non-invasive Features')
    plt.ylabel('Interventional Features')
    plt.xticks(ticks=np.arange(len(features)),  # 生成0-35的刻度位置
           labels=features,                 # 使用特征名称作为标签
           rotation=60,                     # 45度倾斜
           ha='right',                      # 标签右对齐
           fontsize=20)                     # 适当缩小字体

    # 调整画布边距（关键参数）
    plt.subplots_adjust(bottom=0.25, left=0.15)  # 下边距增大25%，左边距15%

    # Y轴同理设置（如需）
    plt.yticks(ticks=np.arange(len(features)),
           labels=features,
           fontsize=20)
    plt.tight_layout()
    
    # 保存图片
    img_path = os.path.join(output_folder, f"{sheep_id}_{comparison}.png")
    plt.savefig(img_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"结果已保存: {excel_path}")

def main_analysis(data_folder, output_folder):
    """主分析流程"""
    # 加载数据
    all_data = load_data(data_folder)
    
    # 按羊分组
    analysis_groups = {}
    for (sheep, stype), df in all_data.items():
        analysis_groups.setdefault(sheep, {})[stype] = df
        
    # 执行分析
    for sheep, data in analysis_groups.items():
        required = ['Intervention', 'Left', 'Right']
        if not all(r in data for r in required):
            print(f"{sheep} 数据不完整，跳过分析")
            continue
            
        # 获取数据
        intervention = data['Intervention']
        left = data['Left']
        right = data['Right']
        
        # 分析两种对比
        comparisons = {
            'Intervention_vs_Non-invasive(Cz)': (intervention, left),
            'Intervention_vs_Non-invasive(C1)': (intervention, right)
        }
        
        for comp_name, (src, tgt) in comparisons.items():
            # 计算相关系数矩阵
            corr_matrix = calculate_correlation(src, tgt)
            
            # 保存结果
            save_results(sheep, comp_name, corr_matrix, output_folder)

if __name__ == "__main__":
    # 配置路径
    DATA_FOLDER = "Data\Correlation\Feature_data_excel"  # 替换为实际路径
    OUTPUT_FOLDER = "Data\Correlation\Correlation_Results"
    
    # 执行分析
    main_analysis(DATA_FOLDER, OUTPUT_FOLDER)