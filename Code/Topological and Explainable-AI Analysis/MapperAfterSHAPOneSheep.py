import os
import pandas as pd
import numpy as np
import kmapper as km
from sklearn import preprocessing, cluster, decomposition
from umap import UMAP
import matplotlib.pyplot as plt

# =====================
# 1. 多文件数据加载与标记（10特征版）
# =====================
def load_data_with_labels():
    """加载6个通道数据并添加分层标签"""
    data_paths = [
        ("Data\SHAP\data_AfterSHAPOneSheep/1A.xlsx", "077", "invasive"),
        ("Data\SHAP\data_AfterSHAPOneSheep/1B.xlsx", "077", "non-invasive_left"),
        ("Data\SHAP\data_AfterSHAPOneSheep/1C.xlsx", "077", "non-invasive_right")
    ]
    
    dfs = []
    for path, sheep, signal_type in data_paths:
        # 加载数据并统一列名
        df = pd.read_excel(path)
        
        # 统一特征列名为f0-f9格式（10个特征）
        feature_columns = [f"f{i}" for i in range(10)]  # 修改为10个特征列
        remaining_columns = list(df.columns[10:])       # 处理超出10列的情况
        
        # 重组列名并保留原始额外列
        df.columns = feature_columns + remaining_columns
        
        # 添加元数据
        df = df.iloc[:, :10]  # 确保只取前10个特征列
        df['sheep'] = sheep
        df['signal_type'] = signal_type
        df['time_segment'] = np.arange(len(df))
        
        dfs.append(df)
        print(f"已加载 {path}: 有效特征列={len(feature_columns)}, 总形状={df.shape}")

    combined = pd.concat(dfs, ignore_index=True)
    print("\n合并后数据总形状:", combined.shape)
    return combined

# =====================
# 2. 数据预处理（增强版）
# =====================
# 加载数据
full_data = load_data_with_labels()

# 检查缺失值
nan_check = full_data.isna().sum()
print("\n缺失值统计:")
print(nan_check[nan_check > 0])

# 处理缺失值（删除包含缺失值的行）
initial_count = len(full_data)
full_data = full_data.dropna()
print(f"\n删除缺失值后保留样本数: {len(full_data)}/{initial_count} ({len(full_data)/initial_count:.1%})")

# 分离特征和元数据
meta_cols = ['sheep', 'signal_type', 'time_segment']
features = full_data.drop(meta_cols, axis=1)
meta_data = full_data[meta_cols]

# 数据标准化
scaler = preprocessing.StandardScaler()
scaled_features = scaler.fit_transform(features)

# 检查标准化后的NaN
if np.isnan(scaled_features).any():
    print("\n警告：标准化后数据仍存在NaN！")
    print("NaN分布:", np.isnan(scaled_features).sum(axis=0))
else:
    print("\n标准化验证完成，无NaN值")

# =====================
# 3. Mapper参数配置
# =====================
mapper = km.KeplerMapper()

# # 投影参数（调整UMAP参数增强稳定性）
# projector = UMAP(
#     n_components=2,
#     n_neighbors=50,      # 增加邻域大小
#     min_dist=0.1,        # 减少最小距离
#     random_state=42,
#     metric='euclidean'
# )

# # 聚类参数优化
# clusterer = cluster.HDBSCAN(
#     min_cluster_size=20,
#     cluster_selection_epsilon=0.5
# )

# cover = km.Cover(n_cubes=15, perc_overlap=0.3)

# 投影参数保持不变
projector = UMAP(
    n_components=2,
    n_neighbors=50,
    min_dist=0.1,
    random_state=42,
    metric='euclidean'  # 此处已使用L2距离
)

# 修改聚类器为基于L2的DBSCAN
clusterer = cluster.DBSCAN(
    eps=0.5,            # 邻域半径
    min_samples=20,      # 核心点所需最小样本数
    metric='euclidean'   # 显式指定L2距离
)

cover = km.Cover(n_cubes=15, perc_overlap=0.3)

# =====================
# 4. 构建Mapper图
# =====================
print("\n开始UMAP降维...")
projected = projector.fit_transform(scaled_features)
print("UMAP投影完成，形状:", projected.shape)

print("\n构建Mapper图中...")
graph = mapper.map(
    projected,
    clusterer=clusterer,
    cover=cover,
    remove_duplicate_nodes=True
)
print(f"生成拓扑图包含 {len(graph['nodes'])} 个节点")

# =====================
# 5. 增强型特征分析
# =====================
def enhanced_cluster_analysis(graph, features, meta_data):
    """带数据验证的簇分析"""
    cluster_stats = []
    
    for node_id, members in graph['nodes'].items():
        # 数据验证
        if len(members) == 0:
            continue
            
        try:
            node_data = {
                'node_id': node_id,
                'size': len(members),
                'signal_dist': meta_data.iloc[members]['signal_type'].value_counts(normalize=True).to_dict(),
                'sheep_dist': meta_data.iloc[members]['sheep'].value_counts(normalize=True).to_dict(),
                'time_span': (
                    meta_data.iloc[members]['time_segment'].min(),
                    meta_data.iloc[members]['time_segment'].max()
                ),
                'features_mean': features.iloc[members].mean().to_dict(),
                'features_std': features.iloc[members].std().to_dict()
            }
            cluster_stats.append(node_data)
        except Exception as e:
            print(f"节点 {node_id} 分析出错: {str(e)}")
            continue
    
    return pd.DataFrame(cluster_stats)

print("\n进行簇特征分析...")
cluster_df = enhanced_cluster_analysis(graph, features, meta_data)

# =====================
# 6. 替代可视化方案（修正版）
# =====================
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# 生成节点数据
node_info = []
for node_id, members in graph["nodes"].items():
    # 获取节点元数据
    signal_dist = cluster_df[cluster_df['node_id'] == node_id]['signal_dist'].values[0]
    time_span = cluster_df[cluster_df['node_id'] == node_id]['time_span'].values[0]
    
    node_info.append({
        "position": np.median(projected[members], axis=0),
        "color": signal_dist.get('invasive', 0),
        "size": len(members),          # 实际样本数量
        "display_size": 50 + 100 * np.log1p(len(members)),  # 显示用尺寸
        "time_span": f"{time_span[0]}-{time_span[1]}"
    })

# 准备可视化数据
positions = np.array([ni["position"] for ni in node_info])
colors = [ni["color"] for ni in node_info]
sizes = [ni["display_size"] for ni in node_info]
labels = [
    f"Node {i}\n"
    f"Samples: {node_info[i]['size']}\n"
    f"Time: {node_info[i]['time_span']}"
    for i in range(len(node_info))
]

# 创建可视化
plt.figure(figsize=(15, 12))
scatter = plt.scatter(
    positions[:, 0],
    positions[:, 1],
    c=colors,
    cmap=LinearSegmentedColormap.from_list("invasion", ["blue", "red"]),
    s=sizes,
    alpha=0.7,
    edgecolors='black'
)

# 添加标注（仅标注样本量前20%的节点）
threshold = np.percentile([ni["size"] for ni in node_info], 80)
for i, pos in enumerate(positions):
    if node_info[i]["size"] > threshold:
        plt.text(pos[0], pos[1], labels[i],
                 fontsize=8, ha='center', va='bottom',
                 bbox=dict(facecolor='white', alpha=0.8))

# 添加图例和颜色条
cbar = plt.colorbar(scatter)
cbar.set_label('Invasive Signal Ratio')
plt.title("EEG Signal Topology Analysis (Node Size ≈ Sample Count)")
plt.xlabel("UMAP Dimension 1")
plt.ylabel("UMAP Dimension 2")

# 保存输出
plt.savefig("EEG_Topology_Revised.png", dpi=300, bbox_inches='tight')
print("可视化文件已生成: EEG_Topology_Revised.png")

# =====================
# 7. 结果输出与保存
# =====================
# 保存分析结果
cluster_df.to_excel("Cluster_Analysis_Results.xlsx", index=False)
print("\n分析结果已保存")

# 生成特征报告
print("\n=== 关键特征分布 ===")
# 将每个features_mean字典转换为DataFrame的行
feature_means = pd.DataFrame(cluster_df['features_mean'].tolist()) 
print(feature_means.describe().T.sort_values('mean', ascending=False).head(10))

print("\n=== 运行完成 ===")