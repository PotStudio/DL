from pylab import *
import pandas as pd
import matplotlib.pylab as plt

filePath = ("rain.csv")
dataFile = pd.read_csv(filePath)

describe = dataFile.describe()
print(describe)

array = dataFile.iloc[:, 1:13].values
print(array)
boxplot(array)
plt.xlabel("month")
plt.ylabel(("rain"))
show()
