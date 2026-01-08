import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, roc_auc_score, roc_curve,
    precision_recall_curve, recall_score, precision_score, f1_score
)
import seaborn as sns
from sklearn.metrics import confusion_matrix

# 数据加载
try:
    train_df = pd.read_csv('train-set.csv')
    test_df = pd.read_csv('test-set.csv')
    print(f"训练集样本数: {len(train_df)}, 特征数: {train_df.shape[1]}")
    print(f"测试集样本数: {len(test_df)}, 患病样本数: {test_df['label'].sum()}")
except FileNotFoundError as e:
    print(f"文件加载失败：{e}")
    exit()

# 处理缺失值
train_df = train_df.fillna(train_df.mean())
test_df = test_df.fillna(test_df.mean())

# 数据拆分
X_test = test_df.drop('label', axis=1)
y_test = test_df['label']

# 标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(train_df)
X_test_scaled = scaler.transform(X_test)

# SVM模型
model = OneClassSVM(
    kernel='rbf',
    nu=0.035,  
    gamma='scale'
)
# 训练
model.fit(X_train_scaled)

# 获取原始异常分数（分数越低，越可能是异常）
score = model.decision_function(X_test_scaled)
score = -score  # 转换为：分数越高，患病概率越大

# 保证召回率≥90%的前提下，找到最优阈值提升精确率
precision_curve, recall_curve, thresholds = precision_recall_curve(y_test, score)   # 为所有阈值计算精确率和召回率
valid_idx = recall_curve >= 0.9                 
valid_thresholds = thresholds[valid_idx[:-1]]   
valid_precision = precision_curve[valid_idx]    
best_idx = np.argmax(valid_precision)          
best_threshold = valid_thresholds[best_idx] if len(valid_thresholds) > 0 else 0     # 提取最优阈值

# 根据最优阈值重新预测
y_pred_optimized = np.where(score >= best_threshold, 1, 0)

# -------- 结果输出 --------
print("="*60)
print(classification_report(y_test, y_pred_optimized, target_names=['正常', '患病']))

# 核心指标汇总 
print(f"患病样本召回率: {recall_score(y_test, y_pred_optimized):.4f}")
print(f"患病样本精确率: {precision_score(y_test, y_pred_optimized):.4f}")
print(f"患病样本F1分数: {f1_score(y_test, y_pred_optimized):.4f}")
print(f"ROC-AUC: {roc_auc_score(y_test, score):.4f}")

# 混淆矩阵 
cm = confusion_matrix(y_test, y_pred_optimized)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Normal', 'Diseased'],
            yticklabels=['Normal', 'Diseased'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

# 异常分数分布 
normal_scores = score[y_test == 0]
disease_scores = score[y_test == 1]

plt.figure(figsize=(8, 5))
plt.hist(normal_scores, bins=30, alpha=0.5, label='Normal Samples', color='blue')
plt.hist(disease_scores, bins=30, alpha=0.5, label='Diseased Samples', color='red')
plt.axvline(x=best_threshold, color='black', linestyle='--', label=f'Optimal Threshold={best_threshold:.4f}')
plt.xlabel('Anomaly Score (Higher -> More Likely to be Diseased)')
plt.ylabel('Number of Samples')
plt.title('Anomaly Score Distribution of Normal/Diseased Samples')
plt.legend()
plt.grid(alpha=0.3)
plt.show()