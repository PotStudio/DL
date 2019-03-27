import pandas as pd
import matplotlib.pyplot as plt
from pylab import *

filePath = ("dataTest.csv")
dataFile = pd.read_csv(filePath, header=None, prefix="V")

print(dataFile.head())
print(dataFile.tail())
# describe对数据进行统计学估计
summary = dataFile.describe()
print(summary)

array = dataFile.iloc[:, 10:16].values
# boxplot 箱形图
boxplot(array)
plt.xlabel("Attribute")
plt.ylabel("Score")
show()
