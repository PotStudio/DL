from pylab import *
import pandas as pd
import matplotlib.pylab as plt

filePath = ("dataTest.csv")
dataFile = pd.read_csv(filePath, header=None, prefix="v")

summary = dataFile.describe()
dataFileNormalized = dataFile.iloc[:, 1:6]
for i in range(5):
    mean = summary.iloc[1, i]
    sd = summary.iloc[2, i]

dataFileNormalized.iloc[:, i:(i+1)] = (dataFileNormalized.iloc[:, i:(i+1)] - mean) / sd
array = dataFileNormalized.values
boxplot(array)
plt.xlabel("Atrribute")
plt.ylabel("Score")
show()