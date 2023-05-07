import pandas as pd #导入Pandas，用于数据读取和处理
# 读入房价数据，示例代码中的文件地址为internet链接，读者也可以下载该文件到本机进行读取
df_housing = pd.read_csv("E:\demo\mechineLearning\house.csv") 
# 构造特征数据集
X = df_housing.drop("median_house_value", axis = 1)
# 构造标签数据集
y = df_housing.median_house_value

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# 导入线性回归模型
from sklearn.linear_model import LinearRegression
model = LinearRegression()
# 训练模型
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print('真实房价', y_test)
print('预测房价', y_pred)

# 打分
print("模型评分：", model.score(X_test, y_test))

import matplotlib.pyplot as plt 
plt.scatter(X_test.median_income, y_test, color = 'red')
plt.plot(X_test.median_income, y_pred, color = 'blue', linewidth = 1)
plt.table('Median Income')
plt.table('Median House Value')
plt.show()

