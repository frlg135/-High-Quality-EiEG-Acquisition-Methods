# -*- coding: utf-8 -*-
"""
适配Excel格式的聚类分析完整解决方案（新增主成分载荷分析）
新增功能：
- 主成分载荷分析模块
- 样本级详细信息保存
- 聚类中心保存
- 质量评估指标保存
- 方差分析结果保存
- 簇统计信息保存
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, adjusted_rand_score
from scipy.stats import f_oneway
import seaborn as sns

# ================= 配置参数 =================
FILE_PATHS = [
    r'Data\Correlation\Feature_data_excel\1A.xlsx',
    r'Data\Correlation\Feature_data_excel\1B.xlsx',
    r'Data\Correlation\Feature_data_excel\1C.xlsx'
]

LABEL_MAPPING = {
    'A': ('intervention', 'none'),
    'B': ('noninvasive', 'left'),
    'C': ('noninvasive', 'right')
}

# ================= 数据加载与预处理 =================
def load_and_preprocess():
    """加载并预处理Excel数据"""
    dfs = []
    
    for file in FILE_PATHS:
        filename = file.split('\\')[-1].split('.')[0]
        subject_id = int(filename[0])
        data_code = filename[1]
        
        signal_type, channel = LABEL_MAPPING[data_code]
        subject_name = f'Sheep-{subject_id:02d}'
        
        df = pd.read_excel(file, header=None, engine='openpyxl')
        df['subject'] = subject_name
        df['signal_type'] = signal_type
        df['channel'] = channel
        
        dfs.append(df)
    
    full_df = pd.concat(dfs, ignore_index=True)
    features = full_df.iloc[:, :36].copy()
    meta = full_df[['subject', 'signal_type', 'channel']].copy()
    
    # 分组标准化
    scaler = StandardScaler()
    scaled_features = []
    for _, group in full_df.groupby('subject'):
        scaled = scaler.fit_transform(group.iloc[:, :36])
        scaled_features.append(scaled)
    
    return np.vstack(scaled_features), meta

# ================= 聚类分析 =================
def perform_clustering(features):
    """执行K-means聚类"""
    model = KMeans(n_clusters=6, 
                  init='k-means++',
                  n_init=50,
                  max_iter=1000,
                  random_state=42)
    return model.fit_predict(features), model

# ================= 可视化模块 =================
def visualize_analysis(pca_data, tsne_data, meta):
    """生成可视化图表"""
    sns.set_theme(style="whitegrid", font_scale=1.2)
    plt.rcParams.update({
        'font.family': 'Arial',
        'figure.dpi': 300,
        'axes.labelweight': 'bold'
    })
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 10))
    plot_projection(ax1, pca_data, meta, 'PCA Projection')
    plot_projection(ax2, tsne_data, meta, 't-SNE Projection')
    
    plt.savefig('cluster_results.png', bbox_inches='tight')
    print("\n可视化图表已保存为 'cluster_results.png'")
    plt.close()

def plot_projection(ax, data, meta, title):
    """绘制投影图"""
    style_map = create_style_mapping(meta)
    handles, labels = [], []
    
    for label in meta['combo_label'].unique():
        idx = meta.index[meta['combo_label'] == label]
        color, marker = style_map[label]
        scatter = ax.scatter(
            data[idx, 0], data[idx, 1],
            c=color, marker=marker, s=80,
            edgecolor='w', linewidth=1,
            label=label, alpha=0.8
        )
        handles.append(scatter)
        labels.append(label.replace("\n", " "))
    
    ax.legend(handles, labels, loc='upper right', fontsize=9,
             frameon=True, framealpha=0.9, title='Legend')
    ax.set_title(title, fontsize=14)
    ax.grid(alpha=0.3)

def create_style_mapping(meta):
    """创建可视化样式映射"""
    colors = {'Sheep-01': '#2ecc71'}
    markers = {
        'intervention': 'o',
        'noninvasive_left': '^',
        'noninvasive_right': 'v'
    }
    
    style_map = {}
    for label in meta['combo_label'].unique():
        subject, stype, channel = label.split('\n')
        key = f"{stype}_{channel}" if stype == 'noninvasive' else stype
        style_map[label] = (colors[subject], markers[key])
    return style_map

# ================= 数据保存模块 =================
def save_results(features, clusters, model, meta, pca_data, tsne_data, pca):
    """保存所有分析结果到Excel文件"""
    
    # 1. 样本级详细信息
    samples_df = pd.DataFrame(features, columns=[f'Feature_{i+1}' for i in range(36)])
    samples_df = pd.concat([meta.reset_index(drop=True), samples_df], axis=1)
    samples_df['cluster'] = clusters
    samples_df['PCA_1'] = pca_data[:, 0]
    samples_df['PCA_2'] = pca_data[:, 1]
    samples_df['tSNE_1'] = tsne_data[:, 0]
    samples_df['tSNE_2'] = tsne_data[:, 1]
    samples_df.to_excel('样本级详细信息.xlsx', sheet_name='Samples', index=False)
    
    # 2. 聚类中心
    cluster_centers_df = pd.DataFrame(
        model.cluster_centers_, 
        columns=[f'Feature_{i+1}' for i in range(36)]
    )
    cluster_centers_df.index.name = 'Cluster'
    cluster_centers_df.reset_index().to_excel('聚类中心.xlsx', sheet_name='Cluster_Centers', index=False)
    
    # 3. 质量评估指标
    metrics_df = pd.DataFrame({
        'Silhouette Score': [silhouette_score(features, clusters)],
        'Adjusted Rand Index': [adjusted_rand_score(meta['combo_label'], clusters)],
        'n_iter_': [model.n_iter_],
        'inertia_': [model.inertia_]
    })
    metrics_df.to_excel('质量评估指标.xlsx', sheet_name='Metrics', index=False)
    
    # 4. 方差分析结果
    pca_3 = PCA(n_components=3).fit_transform(features)
    anova_results = []
    for i in range(3):
        groups = [pca_3[clusters == k, i] for k in range(6)]
        f_value, p_value = f_oneway(*groups)
        anova_results.append({
            'PC': f'PC{i+1}',
            'F-value': f_value,
            'P-value': p_value
        })
    pd.DataFrame(anova_results).to_excel('方差分析结果.xlsx', sheet_name='ANOVA', index=False)
    
    # 5. 主成分载荷分析（新增模块）
    loadings_df = pd.DataFrame(
        pca.components_.T,  # 转置矩阵使特征作为行
        columns=[f'PC{i+1}' for i in range(pca.n_components_)],
        index=[f'Feature_{i+1}' for i in range(36)]
    )
    loadings_df.to_excel('主成分载荷.xlsx', sheet_name='Loadings')
    
    # 6. 簇统计信息（修改部分）
    features_df = pd.DataFrame(features, columns=[f'Feature_{i+1}' for i in range(36)])
    features_df['Cluster'] = clusters

    # 新增：统计signal_type和channel的分布
    signal_type_ratios = pd.crosstab(clusters, meta['signal_type'], normalize='index').add_prefix('signal_type_')*100
    channel_ratios = pd.crosstab(clusters, meta['channel'], normalize='index').add_prefix('channel_')*100

    # 合并统计信息
    stats = pd.concat([
        features_df.groupby('Cluster').mean().add_suffix('_mean'),
        features_df.groupby('Cluster').std().add_suffix('_std'),
        pd.crosstab(clusters, meta['subject'], normalize='index').add_suffix('_ratio')*100,
        signal_type_ratios,
        channel_ratios
    ], axis=1).reset_index()

    stats['Count'] = pd.Series(clusters).value_counts().sort_index().values
    stats.to_excel('簇统计信息.xlsx', sheet_name='Cluster_Stats', index=False)

# ================= 主程序 =================
if __name__ == "__main__":
    # 数据预处理
    features, meta = load_and_preprocess()
    
    # 创建组合标签
    meta = meta.copy()
    meta['combo_label'] = meta.apply(
        lambda x: f"{x['subject']}\n{x['signal_type']}\n{x['channel']}", axis=1
    )
    
    # 执行聚类
    clusters, model = perform_clustering(features)
    
    # 计算降维结果
    pca = PCA(n_components=2)  # 创建PCA实例
    pca_data = pca.fit_transform(features)  # 保存模型用于提取载荷
    tsne_data = TSNE(perplexity=50).fit_transform(features)
    
    # 可视化分析
    visualize_analysis(pca_data, tsne_data, meta)
    
    # 保存分析结果（新增pca参数）
    save_results(features, clusters, model, meta, pca_data, tsne_data, pca)
    
    print("所有分析结果已保存至Excel文件！")