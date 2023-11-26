import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import seaborn as sns
pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

data = pd.read_csv('/Users/shijunshen/Documents/dataset/GTZAN/features_3_sec.csv')
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

# 创建随机森林分类器对象
rf_classifier = RandomForestClassifier()


# 创建回调函数
def print_results(results):
    mean_score = results.cv_results_['mean_test_score'][results.best_index_]
    params = results.cv_results_['params'][results.best_index_]
    print("Mean Score:", mean_score)
    print("Parameters:", params)
    print("---")

# 定义参数网格
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2']
}

# 创建网格搜索对象
grid_search = GridSearchCV(estimator=rf_classifier, param_grid=param_grid, cv=3, verbose=2)

# 使用训练数据拟合网格搜索对象
grid_search.fit(X_train, y_train['label'])

# 打印最佳参数组合和对应的交叉验证准确率
print("Best Parameters: ", grid_search.best_params_)
print("Cross-validated Accuracy: ", grid_search.best_score_)