import numpy as np

from sklearn.metrics import silhouette_score

from measure.distance import Euclidean


class KMeans:

    def __init__(self, n_clusters=2, distance=None):
        """
        Constructor of K-means clustering
        :param n_clusters: Number of clusters
        :param distance: Distance class to compute the distance
        """

        self.n_clusters = n_clusters

        if distance is None:
            distance = Euclidean(degree=2)
        self.distance = distance
        self.feature = None
        self.cluster = None
        self.best_score = 0

    def fit(self, X, y=None, init=None, cluster_weight=None, sample_weight=None, delta=1e-5, max_iter=100, n_iter=3):
        """
        FIt the model to a K Means clustering
        :param X: Feature
        :param y: Targets for the features to be fitted to. Default to be None. Supposed to be None.
        :param init: Inititalisation values. Default to be None.
        :param cluster_weight: Weighting of each clusters
        :param delta: Minimum movements of centroids to be considered convergent.
        :param max_iter: Maximum iterations for centroid adjustments. Default is 100
        :param n_iter: Number of iterations for clustering. Default is 3. When init is None, n_iter is 1.
        :return: self
        """

        if init is None:
            n_iter = 1
            init = {}

        if not isinstance(X, np.ndarray):
            X = np.array(X)

        if len(X.shape) != 2:
            raise ValueError(f"Data must be given as a two dimensional array. Given {len(X.shape)}.")

        data_num, feature_num = X.shape

        if cluster_weight is None:
            cluster_weight = np.ones(self.n_clusters)

        if not isinstance(cluster_weight, np.ndarray):
            cluster_weight = np.array(cluster_weight)

        if len(cluster_weight) != self.n_clusters:
            raise ValueError(f"Expected weights of shape {self.n_clusters}. Given {len(cluster_weight)}")

        if sample_weight is None:
            sample_weight = np.ones(len(X))
        if len(sample_weight) != len(X):
            raise ValueError(f"Expected weights of length {data_num}. Given {len(sample_weight)}")
        if not isinstance(sample_weight, np.ndarray):
            sample_weight = np.array(sample_weight)
        if len(sample_weight.shape) != 1:
            raise ValueError(f"Invalid shape for sample weight.")

        try:
            centroids = init["centroid"]
            if centroids.shape != (self.n_clusters, feature_num):
                raise ValueError(f"Centroid shape not match ({self.n_clusters}, {feature_num})")
        except KeyError:
            centroids = np.random.rand(self.n_clusters, feature_num)

        expanded_X = np.expand_dims(X, 1)
        expanded_weight = np.expand_dims(sample_weight, 1)

        opt_iter = 0
        cluster_list = np.zeros((n_iter, data_num))
        score_list = np.zeros(n_iter)

        while opt_iter < n_iter:
            conv_iter = 0
            delta_centroid = delta + 1
            cluster = np.zeros(data_num)
            while conv_iter < max_iter and delta_centroid > delta:
                distance_from_centroid = self.distance.fit(expanded_X, centroids)
                distance_from_centroid = distance_from_centroid * expanded_weight

                cluster = np.nanargmin(distance_from_centroid, axis=-1)

                new_centroids = np.zeros(centroids.shape)
                for k in range(self.n_clusters):
                    cluster_index = np.where(cluster == k)[0]
                    cluster_point = X[cluster_index]
                    weighting = expanded_weight[cluster_index]
                    cluster_mean = np.sum(cluster_point*weighting, axis=0)/np.sum(weighting)
                    new_centroids[k] = cluster_mean

                delta_centroid = self.distance.fit(new_centroids, centroids)
                delta_centroid *= cluster_weight
                delta_centroid = np.min(delta_centroid)

                centroids = new_centroids

                conv_iter += 1

            score = self.score(X, cluster)
            score_list[opt_iter] = score
            cluster_list[opt_iter] = cluster

            opt_iter += 1

        best_cluster_index = np.nanargmax(score_list)
        best_cluster = cluster_list[best_cluster_index]
        best_score = score_list[best_cluster_index]

        self.feature = X
        self.cluster = best_cluster
        self.best_score = best_score

        return self

    def fit_predict(self, X, y=None, sample_weight=None, **kwargs):
        """
        Compute clusterings for features
        :param X: Features to cluster
        :param y: Default is None.
        :param kwargs: Optional keyword arguments for fitting.
        :return: clusters
        """

        self.fit(X=X, y=y, sample_weight=sample_weight, **kwargs)

        return self.cluster

    def predict(self, X, sample_weight=None, **kwargs):
        """
        Compute clusterings for features
        :param X: Features to cluster
        :param kwargs: Optional keyword arguments for fitting.
        :return: clusters
        """

        return self.fit_predict(X=X, sample_weight=sample_weight, **kwargs)

    def score(self, X=None, y=None, sample_weight=None):
        """
        Compute silhouette score for clustering
        :param X: Features. Default is None.
        :param y: Clustering. Default is None.
        :return: Silhouette score
        """

        if X is None and y is None:
            if self.feature is None or self.cluster is None:
                raise AttributeError("Model is not fitted.")
            else:
                X, y = self.feature, self.cluster

        return silhouette_score(X=X, labels=y)


class KernelKMeans:
    """
    Kernel K-means Clustering
    """

    def __init__(self, kernel=None, **kwargs):

        """
        Constructor for kernel kmeans clustering
        :param kernel: The kernel to use. Default is None, in which case, the ordinary kmeans is constructed.
        :param kwargs: Optional keyword arguments for kmeans clustering.
        """

        self.kernel = kernel
        self.kmeans_estimator = KMeans(**kwargs)

        self.feature = None
        self.clusters = None
        self.kernel_feature = None

    def fit(self, X, y=None, sample_weight=None):

        self.feature = X
        if self.kernel is not None:
            kernel = self.kernel
            transformed_feature = kernel.fit_transform(X=X)
            self.kernel_feature = transformed_feature
            self.kernel = kernel
        else:
            transformed_feature = X

        kmeans_estimator = self.kmeans_estimator
        kmeans_estimator.fit(X=transformed_feature, sample_weight=sample_weight)
        self.kmeans_estimator = kmeans_estimator

        return None

    def fit_predict(self, X, y=None, sample_weight=None):

        self.fit(X=X, y=y)
        if self.kernel is not None:
            clusters = self.kmeans_estimator.fit_predict(self.kernel_feature, sample_weight=sample_weight)
        else:
            clusters = self.kmeans_estimator.fit_predict(X, sample_weight=sample_weight)
        self.clusters = clusters

        return clusters

    def predict(self, X, sample_weight=None):

        if self.kernel is not None:

            pred = self.kmeans_estimator.predict(X=self.kernel_feature, sample_weight=sample_weight)

        else:

            pred = self.kmeans_estimator.predict(X=X, sample_weight=sample_weight)

        return pred

    def score(self, X, y=None, sample_weight=None):

        if y is None:
            y = self.predict(X, sample_weight=sample_weight)

        if self.kernel is not None:
            X = self.kernel.transform(X)

        silhouette = silhouette_score(X=X, labels=y)

        return silhouette