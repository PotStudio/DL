import pandas as pd
import matplotlib.pyplot as plot


# DataFame是一种二维表
rocksMines = pd.DataFrame([[1, 200, 105, 3, False],
                          [2, 165, 80, 2, False],
                          [3, 184.5, 120, 2, False],
                          [4, 116, 70.8, 1, False],
                          [5, 270, 150, 4, True]])
# pd.DataFrame切片
dataRow1 = rocksMines.iloc[1, 0:3]
dataRow2 = rocksMines.iloc[2, 0:3]
# scatter仅画点，plot画点，并将点用线连起来
plot.scatter(dataRow1, dataRow2)
plot.xlabel("Attribute1")
plot.ylabel("Attribute2")
plot.show()

dataRow3 = rocksMines.iloc[3, 0:3]
plot.scatter(dataRow2, dataRow3)
plot.xlabel("Attribute2")
plot.ylabel("Attribute3")
plot.show()

