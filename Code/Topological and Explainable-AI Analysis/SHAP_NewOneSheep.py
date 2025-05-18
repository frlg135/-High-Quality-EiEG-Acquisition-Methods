import shap
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.svm import SVC
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.preprocessing import StandardScaler
import os

# 配置matplotlib中文字体（如果需要可取消注释）
# plt.rcParams['font.sans-serif'] = ['SimHei']

# 文件路径配置
file_paths = [
    r'Data\SHAP\data\1A.xlsx',   # 羊077 介入式
    r'Data\SHAP\data\1B.xlsx',   # 羊077 非侵入式左
    r'Data\SHAP\data\1C.xlsx'
]

# 数据读取与预处理
X_list = []
y_list = []

for file in file_paths:
    # 读取数据并处理缺失值
    data = pd.read_excel(file, header=None).dropna()
    
    # 分配标签：文件名包含"A"为介入式（1），其余为0
    label = 1 if 'A.xlsx' in file else 0
    
    # 存储数据
    X_list.append(data.values)
    y_list.extend([label] * len(data))

# 合并数据集
X = np.vstack(X_list)
y = np.array(y_list)

# -------------------------- 全特征五折交叉验证 --------------------------
cv = KFold(n_splits=5, shuffle=True, random_state=42)
scaler_full = StandardScaler()
accuracies_full = []

for train_idx, test_idx in cv.split(X):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    # 标准化处理
    X_train_scaled = scaler_full.fit_transform(X_train)
    X_test_scaled = scaler_full.transform(X_test)
    
    # 训练模型
    model = SVC(
        C=10, gamma=1, kernel='rbf',
        probability=True, class_weight='balanced',
        random_state=42
    )
    model.fit(X_train_scaled, y_train)
    
    # 记录准确率
    acc = accuracy_score(y_test, model.predict(X_test_scaled))
    accuracies_full.append(acc)

print("[全特征] 五折交叉验证准确率:", accuracies_full)
print(f"[全特征] 平均准确率: {np.mean(accuracies_full):.4f} ± {np.std(accuracies_full):.4f}\n")

# -------------------------- SHAP特征分析 --------------------------
# 使用完整数据训练解释模型
scaler_shap = StandardScaler()
X_shap_scaled = scaler_shap.fit_transform(X)
explain_model = SVC(
    C=10, gamma=1, kernel='rbf',
    probability=True, class_weight='balanced',
    random_state=42
)
explain_model.fit(X_shap_scaled, y)

# SHAP解释器
explainer = shap.Explainer(explain_model.predict_proba, X_shap_scaled)
shap_values = explainer(X_shap_scaled)

# 生成特征名称
feature_names = [f"Feature_{i}" for i in range(X.shape[1])]

# 筛选Top10特征
shap_values_class1 = shap_values[:, :, 1]
mean_abs_shap = np.mean(np.abs(shap_values_class1.values), axis=0)
top10_idx = np.argsort(-mean_abs_shap)[:10]
top10_features = [feature_names[i] for i in top10_idx]

# 保存筛选结果
pd.DataFrame({'Feature_Index': top10_idx, 'Feature_Name': top10_features}).to_excel('Top10_Features.xlsx', index=False)

# -------------------------- Top10特征五折交叉验证 --------------------------
X_top10 = X[:, top10_idx]
accuracies_top10 = []
scaler_top10 = StandardScaler()

for train_idx, test_idx in cv.split(X_top10):
    X_train, X_test = X_top10[train_idx], X_top10[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    # 标准化处理
    X_train_scaled = scaler_top10.fit_transform(X_train)
    X_test_scaled = scaler_top10.transform(X_test)
    
    # 训练模型
    model = SVC(
        C=10, gamma=1, kernel='rbf',
        probability=True, class_weight='balanced',
        random_state=42
    )
    model.fit(X_train_scaled, y_train)
    
    # 记录准确率
    acc = accuracy_score(y_test, model.predict(X_test_scaled))
    accuracies_top10.append(acc)

print("[Top10特征] 五折交叉验证准确率:", accuracies_top10)
print(f"[Top10特征] 平均准确率: {np.mean(accuracies_top10):.4f} ± {np.std(accuracies_top10):.4f}")

# -------------------------- 保存结果到Excel --------------------------
results_df = pd.DataFrame({
    'Fold': range(1, 6),
    'Full_Features_Accuracy': accuracies_full,
    'Top10_Features_Accuracy': accuracies_top10
})

summary_df = pd.DataFrame({
    'Model': ['Full Features', 'Top10 Features'],
    'Mean_Accuracy': [np.mean(accuracies_full), np.mean(accuracies_top10)],
    'Std_Accuracy': [np.std(accuracies_full), np.std(accuracies_top10)]
})

with pd.ExcelWriter('CV_Results.xlsx') as writer:
    results_df.to_excel(writer, sheet_name='Detailed Results', index=False)
    summary_df.to_excel(writer, sheet_name='Summary', index=False)

# -------------------------- 可视化与模型保存 --------------------------
# 保存最终模型
joblib.dump(explain_model, 'svm_model.pkl')

# 生成SHAP摘要图
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values[:, :, 1], X_shap_scaled, 
                 feature_names=feature_names, show=False)
plt.title("SHAP Feature Importance")
plt.tight_layout()
plt.savefig('SHAP_Summary.png')
plt.close()

# 生成特征依赖图
for idx in top10_idx:
    plt.figure(figsize=(8, 5))
    shap.dependence_plot(idx, shap_values_class1.values, X_shap_scaled,
                        feature_names=feature_names, show=False)
    plt.title(f"Feature {idx} Dependency")
    plt.tight_layout()
    plt.savefig(f'Dependency_Feature_{idx}.png')
    plt.close()

print("\n结果已保存到 CV_Results.xlsx")

# 保存SHAP值
shap_df = pd.DataFrame(shap_values_class1.values, columns=feature_names)
shap_df['True_Label'] = y_test
shap_df.to_excel('SHAP_Values.xlsx', index=False)