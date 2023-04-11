# clustering dataset
# determine k using elbow method

from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import numpy as np
import matplotlib.pyplot as plt

x1 = np.array([3, 1, 1, 2, 1, 6, 6, 6, 5, 6, 7, 8, 9, 8, 9, 9, 8])
x2 = np.array([5, 4, 5, 6, 5, 8, 6, 7, 6, 7, 1, 2, 1, 2, 3, 2, 3])
print(x1)


dataSet = []  # 列表，用来表示，列表中的每个元素也是一个二维的列表；这个二维列表就是一个样本，样本中包含有我们的属性值和类别号。
# 与我们所熟悉的矩阵类似，最终我们将获得N*2的矩阵，每行元素构成了我们的训练样本的属性值和类别号
fileIn = open("./testSet.txt")  # 是正斜杠
for line in fileIn.readlines():
    temp = []
    lineArr = line.strip().split('\t')  # line.strip()把末尾的'\n'去掉
    temp.append(float(lineArr[0]))
    temp.append(float(lineArr[1]))
    dataSet.append(temp)
print(dataSet)
# dataSet.append([float(lineArr[0]), float(lineArr[1])])
fileIn.close()


plt.plot()
plt.xlim([0, 10])
plt.ylim([0, 10])
plt.title('Dataset')
plt.scatter(x1, x2)
plt.show()

# create new plot and data
plt.plot()
X = np.array(list(zip(x1, x2))).reshape(len(x1), 2)
print(X, X.shape[0], len(dataSet))
colors = ['b', 'g', 'r']
markers = ['o', 'v', 's']

# k means determine k
distortions = []
K = range(1, 10)
for k in K:
    kmeanModel = KMeans(n_clusters=k).fit(dataSet)
    kmeanModel.fit(dataSet)
    distortions.append(sum(np.min(
        cdist(dataSet, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / len(dataSet))

# Plot the elbow
plt.plot(K, distortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Distortion')
plt.title('The Elbow Method showing the optimal k')
plt.show()
