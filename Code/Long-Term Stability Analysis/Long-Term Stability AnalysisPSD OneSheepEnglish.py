import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

# 设置全局字体参数
plt.rcParams.update({
    'font.size': 36,
    'axes.titlesize': 36,
    'axes.labelsize': 36,
    'xtick.labelsize': 36,
    'ytick.labelsize': 36,
    'legend.fontsize': 36,
    'font.sans-serif': ['Times New Roman'],
    'axes.unicode_minus': False
})
plt.rcParams['axes.formatter.use_mathtext'] = True

# 定义颜色配置
SCI_COLORS = {
    'invasive': {'line': '#2F5597', 'band': '#5B8DC9'},
    'non_invasive': {'line': '#ED7D31', 'band': '#FFA042'}
}

# 定义通道信息
channels = [
    "Sheep077-Interventional", "Sheep077-Non-invasive_Cz", "Sheep077-Non-invasive_C1",
]

file_paths = [
    "Data/TimeUseful_PSD/PSD_Sheep1_Channel_1.xlsx",
    "Data/TimeUseful_PSD/PSD_Sheep1_Channel_2.xlsx",
    "Data/TimeUseful_PSD/PSD_Sheep1_Channel_3.xlsx"
]

# 验证文件数量
assert len(file_paths) == 3 and len(channels) == 3, "文件或通道定义数量不正确"

# 读取并处理数据
combined_df = pd.DataFrame()
for path, channel in zip(file_paths, channels):
    df = pd.read_excel(path)
    df["Channel"] = channel
    df["Minute"] = np.arange(len(df)) // 60
    combined_df = pd.concat([combined_df, df], ignore_index=True)

# 定义频段顺序
bands = ["delta", "theta", "alpha", "beta", "gamma", "overall"]

# 创建绘图（不显示图例）
for band in bands:
    fig = plt.figure(figsize=(12, 12))
    ax1 = fig.add_subplot(111)
    ax2 = ax1.twinx()
    
    ax2.set_ylim(1, 1000)
    ax2.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax2.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    
    color_mapping = {
        'Interventional': (SCI_COLORS['invasive']['line'], SCI_COLORS['invasive']['band']),
        'Non_invasive': (SCI_COLORS['non_invasive']['line'], SCI_COLORS['non_invasive']['band'])
    }
    
    # 遍历所有通道
    for idx, channel in enumerate(channels):
        ch_type = 'Interventional' if 'Interventional' in channel else 'Non_invasive'
        line_color, band_color = color_mapping[ch_type]
        
        # 获取数据
        channel_df = combined_df[combined_df["Channel"] == channel]
        grouped = channel_df.groupby("Minute")[band]
        means = grouped.mean().values
        cvs = (grouped.std() / grouped.mean() * 100).fillna(0).values
        
        # 绘制主曲线（按类型设置标签）
        label_psd = 'Invasive PSD' if ch_type == 'Interventional' else 'Non-invasive PSD'
        ax1.plot(
            np.arange(20), means,
            color=line_color,
            linewidth=2.5,
            marker='o',
            markersize=16,
            label=label_psd
        )
        
        # 绘制CV柱状图（按类型设置标签）
        bar_width = 0.15
        positions = np.arange(20) + (idx - 2.5)*bar_width
        label_cv = 'Invasive CV' if ch_type == 'Interventional' else 'Non-invasive CV'
        ax2.bar(
            positions, cvs,
            width=bar_width,
            color=band_color,
            alpha=0.4,
            label=label_cv
        )
    
    # 设置轴参数
    ax1.tick_params(axis='both', which='major', length=6, width=1.2)
    ax2.tick_params(axis='y', which='major', length=6, width=1.2)
    ax1.grid(True, alpha=0.3, linestyle='--')
    
    # 设置标题并调整布局
    plt.title(f"{band}Frequency band - PSD and coefficient of variation analysis", pad=25)
    plt.tight_layout()
    plt.subplots_adjust(top=0.90)
    
    # 保存图表（无图例）
    plt.savefig(f"{band}_带CV标注.png", dpi=300, bbox_inches="tight")
    plt.close()

# 生成独立的图例
fig_legend = plt.figure(figsize=(12, 8))
handles = [
    Line2D([], [], color=SCI_COLORS['invasive']['line'], linewidth=2.5, marker='o', markersize=16, label='Invasive PSD'),
    Rectangle((0,0), 1, 1, color=SCI_COLORS['invasive']['band'], alpha=0.4, label='Invasive CV'),
    Line2D([], [], color=SCI_COLORS['non_invasive']['line'], linewidth=2.5, marker='o', markersize=16, label='Non-invasive PSD'),
    Rectangle((0,0), 1, 1, color=SCI_COLORS['non_invasive']['band'], alpha=0.4, label='Non-invasive CV'),
]
labels = [h.get_label() for h in handles]

# 创建并保存图例
legend = fig_legend.legend(
    handles, labels, loc='center', 
    ncol=2, frameon=True, shadow=True, fancybox=True
)
plt.axis('off')
plt.tight_layout()
plt.savefig('Legend_单独图例.png', bbox_inches='tight', dpi=300)
plt.close()

print("处理完成！生成文件：")
for band in bands:
    print(f"- {band}_带CV标注.png")
print("- Legend_单独图例.png")