import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans


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


class KMeansModel:
    def __init__(self, k: int = 2):
        self.k = k
        self.kmeans = KMeans(n_clusters=k, n_init=10)
        self.label_map: dict = {}

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.kmeans.fit(X)
        cluster_labels = self.kmeans.labels_
        self.label_map = {}
        for cluster_id in range(self.k):
            mask = cluster_labels == cluster_id
            if mask.sum() == 0:
                self.label_map[cluster_id] = 0  # default for empty cluster
            else:
                self.label_map[cluster_id] = int(np.bincount(y[mask]).argmax())

    def predict(self, X: np.ndarray) -> np.ndarray:
        cluster_assignments = self.kmeans.predict(X)
        return np.array([self.label_map[c] for c in cluster_assignments], dtype=int)
