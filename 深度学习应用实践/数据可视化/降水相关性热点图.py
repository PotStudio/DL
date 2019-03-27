from pylab import *
import pandas as pd
import matplotlib.pylab as plt

filePath = ("rain.csv")
dataFile = pd.read_csv(filePath)

describe = dataFile.describe()

corMat = pd.DataFrame(dataFile.iloc[1:20, 1:20].corr())

plt.pcolor(corMat)
plt.show()
# 热点图颜色分布比较平均，说明月份之间的降水量之间相关性 不大