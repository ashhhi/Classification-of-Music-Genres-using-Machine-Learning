import pickle

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier



from sklearn.model_selection import train_test_split
import seaborn as sns
pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

data = pd.read_csv('/Users/shijunshen/Documents/dataset/GTZAN/features_30_sec.csv')
class_names = np.unique(data['label'])
X = data.iloc[:, 2:-1]      # drop 'filename' and 'length' column
y = data.iloc[:, [-1]]
col = data.columns

# print(class_names)
# print(data.head())
# print(data.columns)
# print(y)

#划分训练集和验证集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=231)
# print ("训练集统计描述：\n", X_train.shape)
# print ("验证集统计描述：\n", X_test.shape)
# print ("训练集信息：\n", y_train.shape)
# print ("验证集信息：\n", y_test.shape)
# print(type(X_train))    # <class 'pandas.core.frame.DataFrame'>

# # 相关系数热力图
# correlation = X.corr()
# plt.figure(figsize=(10, 10))
# sns.heatmap(correlation, annot=False, cmap='coolwarm', xticklabels=range(0, correlation.shape[0]), yticklabels=range(0, correlation.shape[0]))
# plt.show()

# 创建 Random Forest 分类器
rf_classifier = RandomForestClassifier(n_estimators=300, max_depth=None, min_samples_split=2, min_samples_leaf=1, max_features='sqrt')
# 使用输入数据和分类标签来拟合分类器
rf_classifier.fit(X_train, y_train['label'])

# 保存模型
with open('./rf_30_model.pkl', 'wb') as f:
    pickle.dump(rf_classifier, f)



