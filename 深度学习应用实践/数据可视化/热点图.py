from pylab import *
import pandas as pd
import matplotlib.pylab as plt

filePath = ("dataTest.csv")
dataFile = pd.read_csv(filePath)

describe = dataFile.describe()

# pd.DataFrame建立二维表
# .corr()相关系数矩阵，即给出了任意两个变量之间的相关系数
corMat = pd.DataFrame(dataFile.iloc[1:20, 1:20].corr())
print(corMat)
plt.pcolor(corMat)
plt.show()
