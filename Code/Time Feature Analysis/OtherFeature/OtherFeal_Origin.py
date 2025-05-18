#所有的时域分析都没有滤波

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import pywt
from scipy.signal import butter, filtfilt
from statsmodels.tsa.ar_model import AutoReg

# 滤波函数
def butter_lowpass_filter(data, cutoff, fs, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y

# 小波平滑函数
def wavelet_smoothing(data, wavelet='db1', level=1):
    coeffs = pywt.wavedec(data, wavelet, level=level)
    coeffs[1:] = [pywt.threshold(i, value=np.std(i), mode='soft') for i in coeffs[1:]]
    return pywt.waverec(coeffs, wavelet)

# 自回归系数函数
def autoregressive_coefficients(signal, order=4):
    model = AutoReg(signal, lags=order)
    model_fitted = model.fit()
    return model_fitted.params

# 分形维数（Higuchi 算法）
def fractal_dimension(signal, kmax=5):
    def L(k):
        length = np.zeros(k)
        for m in range(k):
            for j in range(1, int((len(signal) - m) / k)):
                length[m] += abs(signal[m + j * k] - signal[m + (j - 1) * k])
            length[m] *= (len(signal) - 1) / (j * k)
        return np.log(np.mean(length))

    k = np.arange(1, kmax + 1)
    Lk = np.array([L(ki) for ki in k])
    return np.polyfit(np.log(1 / k), Lk, 1)[0]

# 特征提取函数
def extract_features(signal):
    features = {
        "平均绝对值": np.mean(np.abs(signal)),
        "方差": np.var(signal),
        "均方根值": np.sqrt(np.mean(signal**2)),
        "对数检测器": np.exp(np.mean(np.log(np.abs(signal) + np.finfo(float).eps))),
        "波形长度": np.sum(np.abs(np.diff(signal))),
        "标准差": np.std(signal),
        "绝对标准差差分": np.std(np.diff(signal)),
        "分形维数": fractal_dimension(signal),
        "最大分形长度": np.log(np.sum(np.abs(np.diff(signal)))),
        "肌肉百分比率": np.sum(signal > 2 * np.mean(signal)) / len(signal),
        "积分EMG": np.sum(np.abs(signal)),
        "简单平方EMG": np.sum(signal**2),
        "过零率": ((signal[:-1] * signal[1:]) < 0).sum(),
        "斜率变化率": np.sum(np.diff(np.sign(np.diff(signal))) != 0),
        "威尔逊幅值": np.sum(np.abs(np.diff(signal)) > 0.01),
        "自回归系数": autoregressive_coefficients(signal).tolist()
    }
    return features

# 处理每个文件夹
def process_folder(folder_path, output_prefix, cutoff=40, fs=256):
    for channel_idx in range(3):
        features_list = []
        
        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)
            
            # 读取 Excel 文件
            df = pd.read_excel(file_path)
            signal = df.iloc[:, channel_idx].values
            
            # 滤波和小波平滑
            filtered_data = butter_lowpass_filter(signal, cutoff, fs)
            smoothed_data = wavelet_smoothing(signal)
            
            # 提取特征
            features = extract_features(smoothed_data)
            features_list.append(features)
        
        # 将所有特征转为DataFrame并保存到Excel
        features_df = pd.DataFrame(features_list)
        output_path = f'{output_prefix}_channel_{channel_idx + 1}_features.xlsx'
        features_df.to_excel(output_path, index=False)

# 定义文件夹路径
folder_paths = ['Data/Sheep1', 'Data/Sheep2']

# 处理每个文件夹中的数据
for i, folder_path in enumerate(folder_paths):
    output_prefix = f'Sheep_{i+1}'
    process_folder(folder_path, output_prefix)
