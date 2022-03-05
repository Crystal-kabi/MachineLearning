from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


class KernelKMeans:

    def __init__(self, kernel=None, **kwargs):

        self.kernel = kernel
        self.kmeans_estimator = KMeans(**kwargs)

        self.feature = None
        self.clusters = None
        self.kernel_feature = None

    def fit(self, X, y=None):

        self.feature = X
        if self.kernel is not None:
            kernel = self.kernel
            transformed_feature = kernel.fit_transform(X=X)
            self.kernel_feature = transformed_feature
            self.kernel = kernel
        else:
            transformed_feature = X

        kmeans_estimator = self.kmeans_estimator
        kmeans_estimator.fit(X=transformed_feature)
        self.kmeans_estimator = kmeans_estimator

        return None

    def fit_predict(self, X, y=None):

        self.fit(X=X, y=y)
        if self.kernel is not None:
            clusters = self.kmeans_estimator.fit_predict(self.kernel_feature)
        else:
            clusters = self.kmeans_estimator.fit_predict(X)
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