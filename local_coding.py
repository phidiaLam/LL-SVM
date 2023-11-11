import numpy as np
from sklearn.cluster import KMeans


class LocalCoding:
    def __init__(self, x_value, y_value, anchor_number):
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
        reciprocal_distances_sum = np.sum(1 / (distances + 0.000000001))
        similarities = (1 / (distances + 0.000000001)) / reciprocal_distances_sum

        return similarities.reshape(-1, 1)
