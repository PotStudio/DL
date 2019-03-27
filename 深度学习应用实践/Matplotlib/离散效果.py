from random import uniform

import pandas as pd
import matplotlib.pyplot as plt

filepath = ("dataTest.csv")
datas = pd.read_csv(filepath)

target = []

for i in range(200):
    if datas.iat[i, 10] >= 7:
        target.append(1.0 + uniform(-0.3, 0.3))
    else:
        target.append(0.0 + uniform(-0.3, 0.3))

dataRow = datas.iloc[0:200, 10]
plt.scatter(dataRow, target)
plt.xlabel("Attribute")
plt.ylabel("target")
plt.show()
