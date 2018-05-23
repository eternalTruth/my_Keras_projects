import numpy as np


class Perceptron(object):
    """
    eta:学习率
    n_iter:权重向量的训练次数
    w_：神经分叉权重向量
    errors_：用于记录神经元判断出错次数
    """

    def __init__(self, eta=0.01, n_iter=10):
        self.eta = eta;
        self.n_iter = n_iter
        pass

    def fit(self, X, y):
        """
        输入训练数据，培养神经元，x输入样本向量，y对应样本分类

        X：shape[n_samples, n_features]
        X:[[1,2,3],[4,5,6]]
        n_samples:2
        n_featyres:3

        y:[1, -1]
        """
        """
        初始化权重向量为0
        加1是因为前面算法提到的w0,也就是步调函数阈值
        """
        self.w_ = np.zeros(1 + X.shape[1]);
        self.errors_ = []

        for _ in range(self.n_iter):
            errors = 0
            """
            X:[[1,2,3],[4,5,6]]
            y:[1,-1]
            zip(X,y) = [[1,2,3, 1].[4,5,6 -1]]
            """
            for xi, target in zip(X, y):
                """
                update = η * (y - y')
                eta:学习率
                """
                update = self.eta * (target - self.predict(xi))
                """
                xi是一个向量
                update * xi 等价于：
                [▽w(1) = X[1]*update, ▽w(2) = x[2]*update, ▽w(3) = x[3]*update
                """
                self.w_[1:] += update * xi  # [1:]表示忽略第0个元素，从第一个元素开始
                self.w_[0] += update;

                errors += int(update != 0.0)
                self.errors_.append(errors)

    def net_input(self, X):
        """
        z=w0*1 + w1*x1 + ...+ wn*xn
        """
        """
        np.dot()是矩阵点积运算，
        例如np.dot([1,2,3],[4,5,6]) = 1*4 + 2*5 + 3*6 = 32
        """
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, -1)


file = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
import pandas as pd

df = pd.read_csv(file, header=None)  # None为了说明CSV第一行不是头，而是有用的数据
df.head(10)

import matplotlib.pyplot as plt
import numpy as np

y = df.loc[0:100, 4].values  # 将前100行的第4列抽取出来
y = np.where(y == 'Iris-setosa', -1, 1)  # 一样就转成-1，不一样就转成1

X = df.iloc[0:100, [0, 2]].values  # 抽取第0列和第2列
plt.scatter(X[:50, 0], X[:50, 1], color='red', marker='o', label='setosa')
plt.scatter(X[50:100, 0], X[50:100, 1], color='blue', marker='x', label='versicolor')
plt.xlabel('花瓣长度', fontproperties='SimHei')
plt.ylabel('花径长度', fontproperties='SimHei')
plt.legend(loc='upper left')
plt.show()

ppn = Perceptron(eta=0.1, n_iter=10)
ppn.fit(X, y)
plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('错误分类次数', fontproperties='SimHei')
plt.show()

from matplotlib.colors import ListedColormap


def plot_decision_regions(X, y, classifier, resolution=0.02):
    marker = ('s', 'x', 'o', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max()
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max()

    print(x1_min, x2_max)
    print(x2_min, x2_max)

    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))


plot_decision_regions(X, y, ppn, resolution=0.02)
