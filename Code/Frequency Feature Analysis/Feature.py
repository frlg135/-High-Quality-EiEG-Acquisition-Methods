import os
import pandas as pd
import numpy as np
from scipy.signal import welch
from scipy.stats import kurtosis, skew

# 定义计算特征的函数
def calculate_features(signal, fs=256, nperseg=256):
    frequencies, psd = welch(signal, fs=fs, nperseg=nperseg)
    
    Pmax = np.max(psd)
    Fmax = frequencies[np.argmax(psd)]
    MeanPower = np.mean(psd)
    TotalPower = np.sum(psd)
    MeanFrequency = np.sum(frequencies * psd) / np.sum(psd)
    MedianFrequency = frequencies[np.argmin(np.abs(np.cumsum(psd) - 0.5 * np.sum(psd)))]
    StdDev = np.sqrt(np.mean((psd - np.mean(psd)) ** 2))
    SpectralMoment1 = np.sum(frequencies * psd) / np.sum(psd)
    SpectralMoment2 = np.sum((frequencies ** 2) * psd) / np.sum(psd)
    SpectralMoment3 = np.sum((frequencies ** 3) * psd) / np.sum(psd)
    KurtosisValue = kurtosis(psd)
    SkewnessValue = skew(psd)
    autocorr_coeffs = np.correlate(signal, signal, mode='full') / len(signal)
    autocorr_coeffs = autocorr_coeffs[autocorr_coeffs.size // 2:][:3]
    
    return [Pmax, Fmax, MeanPower, TotalPower, MeanFrequency, MedianFrequency, 
            StdDev, SpectralMoment1, SpectralMoment2, SpectralMoment3, 
            KurtosisValue, SkewnessValue] + list(autocorr_coeffs)

# 文件夹路径
folders = ['Data/Sheep1', 'Data/Sheep2']
output_files = ['Sheep1_Channel1_Features.xlsx', 'Sheep1_Channel2_Features.xlsx', 'Sheep1_Channel3_Features.xlsx',
                'Sheep2_Channel1_Features.xlsx', 'Sheep2_Channel2_Features.xlsx', 'Sheep2_Channel3_Features.xlsx']

# 初始化存储结果的列表
results = [[] for _ in range(6)]

# 遍历文件夹和文件
for folder_idx, folder in enumerate(folders):
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        df = pd.read_excel(file_path, header=None)
        
        # 对每个通道计算特征
        for channel_idx in range(3):
            signal = df.iloc[:, channel_idx].values
            features = calculate_features(signal)
            results[folder_idx * 3 + channel_idx].append(features)

# 将结果保存到Excel文件
for i, result in enumerate(results):
    result_df = pd.DataFrame(result, columns=['Pmax', 'Fmax', 'MeanPower', 'TotalPower', 'MeanFrequency', 
                      'MedianFrequency', 'StdDev', 'SpectralMoment1', 'SpectralMoment2', 
                      'SpectralMoment3', 'Kurtosis', 'Skewness', 'AutocorrCoeff1', 
                      'AutocorrCoeff2', 'AutocorrCoeff3'])
    result_df.to_excel(output_files[i], index=False)
    print(f"Saved {output_files[i]}")
