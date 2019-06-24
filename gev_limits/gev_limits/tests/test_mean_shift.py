from sklearn.cluster import MeanShift
import numpy as np


def test_mean_shift():
    np.random.seed(1)
    xs = np.random.uniform(size=100)
    ys = np.random.uniform(size=100)
    bandwidth = 0.3
    clustering = MeanShift(bandwidth=bandwidth, cluster_all=False)
    clustering.fit(np.array([xs ,ys]).T)
    cluster_mask = clustering.labels_ > -1
    cluster_keys = np.sort(list(set(clustering.labels_[cluster_mask])))
    cluster_counts = []
    for key in cluster_keys:
        cluster_counts.append(np.sum(clustering.labels_ == key))
    largest_cluster = cluster_keys[np.argmax(cluster_counts)]
    best_pos = clustering.cluster_centers_[largest_cluster]