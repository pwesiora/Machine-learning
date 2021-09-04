from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)
print(df.tail())

y = df.iloc[:, 4].values
x = df.iloc[:, 0:3].values
x = x[0:150, 0:2]
y = y[0:150]
y = np.where(y == 'Iris-setosa', 1, -1)


class Perceptron:
    def __init__(self, learning_rate=0.1):
        self.misclassified = []
        self.learning_rate = learning_rate
        self._b = 0.0
        self._w = None

    def fit(self, x: np.array, y: np.array, n_iter=10):
        self._b = 0.0
        self._w = np.zeros(x.shape[1])

        for _ in range(n_iter):
            errors = 0
            for xi, yi in zip(x, y):
                update = self.learning_rate * (yi - self.predict(xi))
                self._b += update
                self._w += update * xi
                errors += int(update != 0.0)

            self.misclassified.append(errors)

    def f(self, x: np.array) -> float:
        return np.dot(x, self._w) + self._b

    def predict(self, x: np.array):
        return np.where(self.f(x) >= 0, 1, -1)


x[:, 0] = (x[:, 0] - x[:, 0].mean()) / x[:, 0].std()
x[:, 1] = (x[:, 1] - x[:, 1].mean()) / x[:, 1].std()

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

percept = Perceptron(learning_rate=0.01)
percept.fit(x_train, y_train)

plt.plot(range(1, len(percept.misclassified) + 1), percept.misclassified, marker='o')
plt.xlabel('Epoch')
plt.ylabel('Number of misclassifications')
plt.show()


def plot_decision_regions(x, y):
    resolution = 0.001

    markers = ('o', '*')
    cmap = ListedColormap(('red', 'green'))
    x1_min, x1_max = x[:, 0].min() - 0.5, x[:, 0].max() + 0.5
    x2_min, x2_max = x[:, 1].min() - 0.5, x[:, 1].max() + 0.5
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = percept.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx, c1 in enumerate(np.unique(y)):
        plt.scatter(x=x[y == c1, 0],
                    y=x[y == c1, 1],
                    alpha=0.8,
                    c=cmap(idx),
                    marker=markers[idx],
                    label=c1)
    plt.show()


plot_decision_regions(x_test, y_test)

from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import seed
import pandas as pd

df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)
print(df.tail())

y = df.iloc[:, 4].values
x = df.iloc[:, 0:3].values
x = x[0:150, 0:2]
y = y[0:150]
y = np.where(y == 'Iris-setosa', 1, -1)
X = df.iloc[0:150, [0, 2]].values

X_std = np.copy(X)
X_std[:, 0] = (X[:, 0] - X[:, 0].mean()) / X[:, 0].std()
X_std[:, 1] = (X[:, 1] - X[:, 1].mean()) / X[:, 1].std()


class Adaline(object):
    def __init__(self, rate=0.02, niter=1000,
                 shuffle=True, random_state=None):
        self.rate = rate
        self.niter = niter
        self.weight_initialized = False
        self.shuffle = shuffle
        if random_state:
            seed(random_state)

    def fit(self, X, y):
        self.initialize_w(X.shape[1])
        self.q = []

        for i in range(self.niter):
            if self.shuffle:
                X, y = self.shuffle_set(X, y)
            q = []
            for xi, target in zip(X, y):
                q.append(self.update(xi, target))
            avg_q = sum(q) / len(y)
            self.q.append(avg_q)
        return self

    def partial_fit(self, X, y):
        if not self.weight_initialized:
            self.initialize_w(X.shape[1])
        if y.ravel().shape[0] > 1:
            for xi, target in zip(X, y):
                self.update(xi, target)
        else:
            self.up
        return self

    def shuffle_set(self, X, y):
        r = np.random.permutation(len(y))
        return X[r], y[r]

    def initialize_w(self, m):
        self.weight = np.zeros(1 + m)
        self.weight_initialized = True

    def update(self, xi, target):
        output = self.net_input(xi)
        error = (target - output)
        self.weight[1:] += self.rate * xi.dot(error)
        self.weight[0] += self.rate * error
        q = 0.5 * error ** 2
        return q

    def net_input(self, X):
        return np.dot(X, self.weight[1:]) + self.weight[0]

    def activation(self, X):
        return self.net_input(X)

    def predict(self, X):
        return np.where(self.activation(X) >= 0.0, 1, -1)


ada = Adaline(niter=1000, rate=0.01, random_state=1)

ada.fit(X_std, y)

plt.plot(range(1, len(ada.q) + 1), ada.q, marker='o', label='batch=1')
plt.xlabel('Epochs')
plt.ylabel('Quality')
plt.show()


