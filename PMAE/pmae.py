import numpy as np
import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import argparse
import logging
from datetime import datetime

# 解析命令行参数
parser = argparse.ArgumentParser(description='Run XGBoost with different weight exponents.')
parser.add_argument('exponent', type=float, help='Exponent value for weight calculation (e.g., 0.1, 0.2, ..., 0.9)')
args = parser.parse_args()

# 设置日志文件
logging.basicConfig(
    filename=f'experiment_log_{args.exponent}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log',
    level=logging.INFO,
    format='%(asctime)s - %(message)s'
)


# 读取数据集
df = pd.read_csv("./to_feaData_-.csv")

# 划分特征和标签
features = df.iloc[:, :-1]
label = df.iloc[:, -1]

# 将字符串标签转换为整数编码
label_encoder = LabelEncoder()
label_encoded = label_encoder.fit_transform(label)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    features, label_encoded, test_size=0.2, stratify=label_encoded, random_state=42
)

# 统计每个类别的样本数量
class_counts = Counter(y_train)
total_samples = len(y_train)

# 计算每个类别的权重
exponent = args.exponent
class_weights = {cls: (1 / count) ** exponent for cls, count in class_counts.items()}

# 为每个样本分配权重
sample_weights = np.array([class_weights[cls] for cls in y_train])

# 初始化 XGBoost 分类器
model = xgb.XGBClassifier(
    objective="multi:softmax",  # 多分类任务
    num_class=len(np.unique(label_encoded)),  # 类别数量
    eval_metric="mlogloss",  # 多分类任务的评估指标
    random_state=30,  # 随机种子
    n_estimators=100,  # 树的数量
    max_depth=6,  # 树的最大深度
    learning_rate=0.1  # 学习率
)

# 训练模型，传入样本权重
model.fit(X_train, y_train, sample_weight=sample_weights)

# 预测
y_pred = model.predict(X_test)

# 将预测结果转换回字符串标签
y_pred_labels = label_encoder.inverse_transform(y_pred)
y_test_labels = label_encoder.inverse_transform(y_test)

# 计算 Micro 和 Macro 指标
micro_precision = precision_score(y_test, y_pred, average="micro")
micro_recall = recall_score(y_test, y_pred, average="micro")
micro_f1 = f1_score(y_test, y_pred, average="micro")

macro_precision = precision_score(y_test, y_pred, average="macro")
macro_recall = recall_score(y_test, y_pred, average="macro")
macro_f1 = f1_score(y_test, y_pred, average="macro")

# 输出结果到日志文件
logging.info(f"Exponent: {exponent}")
logging.info(f"Micro Precision: {micro_precision:.4f}")
logging.info(f"Micro Recall: {micro_recall:.4f}")
logging.info(f"Micro F1: {micro_f1:.4f}")
logging.info(f"Macro Precision: {macro_precision:.4f}")
logging.info(f"Macro Recall: {macro_recall:.4f}")
logging.info(f"Macro F1: {macro_f1:.4f}")
logging.info("\nDetailed Classification Report:")
logging.info(classification_report(y_test_labels, y_pred_labels, digits=4))