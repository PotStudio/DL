from pylab import *
import pandas as pd
import matplotlib.pylab as plt

filePath = ("dataTest.csv")
dataFile = pd.read_csv(filePath, header=None, prefix="v")

describe = dataFile.describe()
minRings = -1
maxRings = 99
nrow = 10
for i in range(nrow):
    dataRow = dataFile.iloc[i, 1:10]
    lableColor = (dataFile.iloc[i, 10] - minRings) / (maxRings - minRings)
    print(lableColor)
    # matplotlib.cm是matplotlib库中内置的彩色映射函数
    dataRow.plot(color=plt.cm.RdYlBu(lableColor), alpha=0.5)
plt.xlabel("Attribute")
plt.ylabel("Score")
show()