import numpy as np
from sklearn.cluster import KMeans


class LocalCoding:
    def __init__(self, x_value, y_value, anchor_number, nearest_number):
        self.nearest_number = nearest_number
        x_pos = x_value[y_value == 1]
        x_neg = x_value[y_value == -1]
        pos_number = round(anchor_number / x_value.shape[0] * x_pos.shape[0])
        pos_number = 1 if pos_number == 0 else pos_number
        neg_number = anchor_number - pos_number
        kmeans_pos = KMeans(n_clusters=int(pos_number), n_init=10)
        kmeans_pos.fit(x_pos)
        if neg_number != 0:
            kmeans_neg = KMeans(n_clusters=int(neg_number), n_init=10)
            kmeans_neg.fit(x_neg)
            self.anchor = np.vstack((kmeans_pos.cluster_centers_, kmeans_neg.cluster_centers_))
        else:
            self.anchor = kmeans_pos.cluster_centers_

    def get_anchor(self):
        return self.anchor

    def fit(self, x):
        distances = np.linalg.norm(self.anchor - x, axis=1)
        min_indices = np.arange(distances.shape[0])
        if distances.shape[0] > self.nearest_number:
            min_indices = np.argpartition(distances[0:], self.nearest_number)[:self.nearest_number]
        min_values = distances[min_indices]
        reciprocal_values = 1 / min_values
        normalized_values = (reciprocal_values - np.min(reciprocal_values)) / (
                    np.max(reciprocal_values) - np.min(reciprocal_values))
        similarities = np.zeros_like(distances)
        similarities[min_indices] = normalized_values

        return similarities.reshape(-1, 1)
