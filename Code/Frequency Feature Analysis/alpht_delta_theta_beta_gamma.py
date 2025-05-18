import h5py
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

# 定义处理函数
def process_and_save_psd(input_file, output_folder, fs=256.0):
    # 打开HDF5文件
    with h5py.File(input_file, 'r') as f:
        # 读取数据集
        data = f['dataset'][:]

    N = len(data)  # 信号长度

    # 使用FFT计算功率谱
    f = np.fft.rfftfreq(N, 1/fs)  # 单边频率
    Y = np.fft.rfft(data) / N  # FFT并归一化
    P2 = np.abs(Y)**2  # 功率谱（幅度谱的平方）

    # 定义五个频率范围
    freq_ranges = {
        'Delta': (0.1, 3),
        'Theta': (4, 7),
        'Alpha': (8, 13),
        'Beta': (13, 28),
        'Gamma': (28, 100)
    }

    # 遍历频率范围并保存数据到Excel文件
    for name, (low, high) in freq_ranges.items():
        # 找到对应频率范围的索引
        indices = np.where((f >= low) & (f <= high))[0]
        # 提取对应频率范围和强度值
        freq_values = f[indices]
        power_values = 10 * np.log10(P2[indices])  # 转换为dB

        # 创建一个DataFrame
        df = pd.DataFrame({'Frequency (Hz)': freq_values, 'Power (dB/Hz)': power_values})

        # 创建输出文件夹
        os.makedirs(output_folder, exist_ok=True)

        # 保存DataFrame到Excel文件
        file_name = os.path.join(output_folder, f"{name}_Power_Spectrum.xlsx")
        df.to_excel(file_name, index=False)
        print(f"Saved {file_name}")

    # 绘制整个功率谱图
    plt.figure(figsize=(10, 6))
    plt.plot(f, 10*np.log10(P2))
    plt.xlabel('Freq (Hz)')
    plt.ylabel('Power/Frequency (dB/Hz)')
    plt.grid()
    plt.title(f'Power Spectrum of the EEG Signal ({os.path.basename(input_file)})')

    # 绘制频率范围的阴影区域（可选）
    for name, (low, high) in freq_ranges.items():
        plt.axvspan(low, high, facecolor=f'C{ord(name[0])-65}', alpha=0.2, label=name)

    # 添加图例
    plt.legend()
    plt.xlim([0, fs/2])  # 限制x轴范围为0到采样频率的一半
    plt.show()

    # 保存整个频率范围的功率谱密度到Excel文件
    full_spectrum_df = pd.DataFrame({'Frequency (Hz)': f, 'Power (dB/Hz)': 10*np.log10(P2)})
    full_spectrum_file_name = os.path.join(output_folder, "Full_Power_Spectrum.xlsx")
    full_spectrum_df.to_excel(full_spectrum_file_name, index=False)
    print(f"Saved {full_spectrum_file_name}")

# 文件路径列表
file_paths = [
    'Data/Sheep_Origin1_1.h5',
    'Data/Sheep_Origin1_2.h5',
    'Data/Sheep_Origin1_3.h5',
    'Data/Sheep_Origin2_1.h5',
    'Data/Sheep_Origin2_2.h5',
    'Data/Sheep_Origin2_3.h5'
]

# 对应的输出文件夹路径
output_folders = [
    'Data/PSD_Sheep_Origin1_1',
    'Data/PSD_Sheep_Origin1_2',
    'Data/PSD_Sheep_Origin1_3',
    'Data/PSD_Sheep_Origin2_1',
    'Data/PSD_Sheep_Origin2_2',
    'Data/PSD_Sheep_Origin2_3'
]

# 处理每个文件并提取PSD
for input_file, output_folder in zip(file_paths, output_folders):
    process_and_save_psd(input_file, output_folder)
