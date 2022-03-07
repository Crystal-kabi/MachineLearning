import numpy as np

from measure.distance import Euclidean


class Silhouette:

    def __init__(self, distance=None):
        """
        Constructor of a Silhouette scorer
        :param distance: Distance class used to compute distance
        """

        if distance is None:
            distance = Euclidean(degree=2)
        self.distance = distance

    def single_point_silhouette(self, x_index, X, y, sample_weight=None):
        """
        Compute the silhouette score of a single point
        :param x_index: Index of the point
        :param X: Features
        :param y: Labels
        :param sample_weight: Weighting of each sample
        :return: Silhouette score of the point
        """

        if not isinstance(X, np.ndarray):
            X = np.array(X)
        if len(X.shape) != 2:
            raise ValueError(f"Expected shape of dim 2 for X. Given: {len(X.shape)}")

        if not isinstance(y, np.ndarray):
            y = np.array(y)
        if len(y.shape) != 1:
            raise ValueError(f"Expected shape of dim 1 for y. Given: {len(y.shape)}")

        if sample_weight is None:
            sample_weight = np.ones(len(X))
        if not isinstance(sample_weight, np.ndarray):
            sample_weight = np.array(sample_weight)
        if len(sample_weight.shape) != 1:
            raise ValueError(f"Expected shape of dim 1 for sample weight. Given: {len(sample_weight.shape)}")

        all_clusters = np.unique(y)

        self_element = X[x_index]
        self_cluster = y[x_index]

        corrected_weight = sample_weight.copy()
        corrected_weight[x_index] = 0

        distance_array = self.distance.fit(self_element, x)

        a = 0
        b_array = np.zeros(len(all_clusters)-1)

        other_cnt = 0
        for i in all_clusters:

            cluster_index = np.where(y == i)[0]

            if len(cluster_index) == 1:

                mean_distance = 0

            else:

                cluster_distance = distance_array[cluster_index]

                cluster_weight = corrected_weight[cluster_index]

                mean_distance = np.sum(cluster_weight*cluster_distance) / np.sum(cluster_weight)

            if i == self_cluster:

                a = mean_distance

            else:

                b_array[other_cnt] = mean_distance
                other_cnt += 1

        b = np.min(b_array)

        point_silhouette = (b - a) / max(a, b)

        return point_silhouette

    def silhouette_score(self, X, y, sample_weight=None):

        """
        Compute the mean of all sample silhouette scores
        :param X: Features
        :param y: Labels
        :param sample_weight: Weighting of each sample
        :return: Mean silhouette score of the data
        """

        if not isinstance(X, np.ndarray):
            X = np.array(X)
        if len(X.shape) != 2:
            raise ValueError(f"Expected shape of dim 2 for X. Given: {len(X.shape)}")

        if not isinstance(y, np.ndarray):
            y = np.array(y)
        if len(y.shape) != 1:
            raise ValueError(f"Expected shape of dim 1 for y. Given: {len(y.shape)}")

        if sample_weight is None:
            sample_weight = np.ones(len(X))
        if not isinstance(sample_weight, np.ndarray):
            sample_weight = np.array(sample_weight)
        if len(sample_weight.shape) != 1:
            raise ValueError(f"Expected shape of dim 1 for sample weight. Given: {len(sample_weight.shape)}")

        score = 0

        for i in range(len(X)):

            point_silhouette = self.single_point_silhouette(x_index=i, X=X, y=y, sample_weight=sample_weight)
            point_silhouette *= sample_weight[i]
            score += point_silhouette

        score /= np.sum(sample_weight)

        return score


if __name__ == '__main__':

    x = np.random.rand(50, 10)
    labels = np.random.randint(0, 3, 50)
    weight = np.random.rand(50)

    silhouette_scorer = Silhouette()

    s = silhouette_scorer.single_point_silhouette(x_index=4, X=x, y=labels)
    mean_s = silhouette_scorer.silhouette_score(X=x, y=labels)
