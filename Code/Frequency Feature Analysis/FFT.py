import h5py
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# 读取和标准化数据
def process_data(file_path):
    with h5py.File(file_path, 'r') as f:
        data = f['dataset'][:]
        
        # 标准化
        scaler = StandardScaler()
        normalized_data = scaler.fit_transform(data.reshape(-1, 1)).flatten()
        
        return normalized_data

# 计算FFT
def compute_fft(signal, fs):
    n = len(signal)
    fft_values = np.fft.fft(signal)
    fft_values = np.abs(fft_values[:n//2])  # 只取一半频率（正频率）
    freqs = np.fft.fftfreq(n, d=1/fs)[:n//2]
    
    return freqs, fft_values

# 绘制FFT曲线
def plot_fft(freqs, fft_values, title):
    plt.figure(figsize=(10, 6))
    plt.plot(freqs, fft_values)
    plt.title(title)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.show()

# 文件路径列表
file_paths = [
    'Data/Sheep_Origin1_1.h5',
    'Data/Sheep_Origin1_2.h5',
    'Data/Sheep_Origin1_3.h5',
    'Data/Sheep_Origin2_1.h5',
    'Data/Sheep_Origin2_2.h5',
    'Data/Sheep_Origin2_3.h5'
]

# 采样频率（根据实际数据的采样频率设定）
fs = 256

# 处理每个文件并保存FFT数据到Excel和绘制FFT曲线
for i, file_path in enumerate(file_paths):
    # 标准化处理后的数据
    processed_data = process_data(file_path)
    
    # 计算FFT
    freqs, fft_values = compute_fft(processed_data, fs)
    
    # 将频率和幅值数据保存到DataFrame
    fft_df = pd.DataFrame({
        'Frequency (Hz)': freqs,
        'Amplitude': fft_values
    })
    
    # 保存到Excel文件
    output_path = f'Data/FFT_File_{i+1}.xlsx'
    fft_df.to_excel(output_path, index=False)
    print(f'Saved FFT data to {output_path}')
    
    # 绘制FFT曲线
    title = f'FFT Spectrum for File {i+1}'
    plot_fft(freqs, fft_values, title)
