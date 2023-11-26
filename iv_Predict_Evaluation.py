import pickle

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

data = pd.read_csv('/Users/shijunshen/Documents/dataset/GTZAN/features_3_sec.csv')
class_names = np.unique(data['label'])
X = data.iloc[:, 2:-1]      # drop 'filename' and 'length' column
y = data.iloc[:, [-1]]
col = data.columns

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=231)


with open('./rf_3_model.pkl', 'rb') as f:
    rf_classifier = pickle.load(f)


# 对新数据进行分类预测
new_data = X_test
y_pred = rf_classifier.predict(new_data)
y_true = np.array(y_test['label'])

different_items = [(item1, item2) for idx, (item1, item2) in enumerate(zip(y_pred, y_true)) if item1 != item2]

different_items = pd.DataFrame(different_items, columns=['prediction', 'ground truth'])
print(different_items.tail())
# 计算准确率
accuracy = accuracy_score(y_true, y_pred)
print("Accuracy：", accuracy)

# 计算精确率
precision = precision_score(y_true, y_pred, average='macro')
print("Precision：", precision)

# 计算召回率
recall = recall_score(y_true, y_pred, average='macro')
print("Recall：", recall)

# 计算F1分数
f1 = f1_score(y_true, y_pred, average='macro')
print("F1 Score：", f1)
