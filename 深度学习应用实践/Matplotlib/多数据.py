import matplotlib.pyplot as plot
import pandas as pd

filePath = ("dataTest.csv")

# 在没有列标题，就是header=None时，给列数据添加前缀
datas = pd.read_csv(filePath, header=None, prefix="V")
# pandas.DataFrame.iat提取制定元素
target = []
for i in range(200):
    if datas.iat[i, 10] >= 7:
        target.append(1.0)
    else:
        target.append(0.0)

dataRow = datas.iloc[0:200, 10]
plot.scatter(dataRow, target)
plot.xlabel("Attribute")
plot.ylabel("target")
plot.show()
