import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB


class KNNModel:
    def __init__(self, k: int = 5):
        self.k = k
        self.knn = KNeighborsClassifier(n_neighbors=k)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.knn.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.knn.predict(X).astype(int)


class DecisionTreeModel:
    def __init__(self):
        self.tree = DecisionTreeClassifier()

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.tree.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.tree.predict(X).astype(int)


class NaiveBayesModel:
    def __init__(self):
        self.nb = GaussianNB()

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.nb.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.nb.predict(X).astype(int)
