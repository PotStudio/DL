import numpy as np

data = np.mat([[1, 200, 105, 3, False], [2, 165, 80, 2, False], [5, 270, 150, 4, True]])

coll = []
for line in data:
    coll.append(line[0, 1])
print(np.sum(coll))
print(np.mean(coll))
# 标准差
print(np.std(coll))
# 方差
print(np.var(coll))
