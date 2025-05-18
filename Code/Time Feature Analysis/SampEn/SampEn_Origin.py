import os
import pandas as pd
import numpy as np
from antropy import sample_entropy
from sklearn.preprocessing import StandardScaler
import pywt

# 标准化和小波平滑函数
def preprocess_data(data, wavelet='db1', level=1):
    # 标准化
    scaler = StandardScaler()
    normalized_data = scaler.fit_transform(data.reshape(-1, 1)).flatten()
    
    # 小波平滑
    coeffs = pywt.wavedec(normalized_data, wavelet, level=level)
    coeffs[1:] = [pywt.threshold(i, value=np.std(i), mode='soft') for i in coeffs[1:]]
    smoothed_data = pywt.waverec(coeffs, wavelet)
    
    return smoothed_data

# 计算文件夹中所有文件的样本熵并保存结果
def process_folder(folder_path, output_prefix):
    channel_1_entropy = []
    channel_2_entropy = []
    channel_3_entropy = []
    
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        
        # 读取 Excel 文件
        df = pd.read_excel(file_path)
        
        # 对三个通道的数据进行处理和计算样本熵
        for i, channel_data in enumerate([df.iloc[:, 0], df.iloc[:, 1], df.iloc[:, 2]]):
            processed_data = preprocess_data(channel_data.values)
            entropy_value = sample_entropy(processed_data)
            
            if i == 0:
                channel_1_entropy.append(entropy_value)
            elif i == 1:
                channel_2_entropy.append(entropy_value)
            elif i == 2:
                channel_3_entropy.append(entropy_value)
    
    # 保存样本熵到 Excel 文件
    pd.DataFrame(channel_1_entropy).to_excel(f'{output_prefix}_channel_1_entropy.xlsx', index=False, header=False)
    pd.DataFrame(channel_2_entropy).to_excel(f'{output_prefix}_channel_2_entropy.xlsx', index=False, header=False)
    pd.DataFrame(channel_3_entropy).to_excel(f'{output_prefix}_channel_3_entropy.xlsx', index=False, header=False)

# 定义文件夹路径
folder_paths = ['Data\Sheep1', 'Data\Sheep2']

# 处理文件夹中的数据
for i, folder_path in enumerate(folder_paths):
    output_prefix = f'Sheep_{i+1}'
    process_folder(folder_path, output_prefix)
