import numpy as np


class Distance:

    def __init__(self):
        """
        General distance class
        """
        pass

    def fit(self, x, y):
        """
        Compute the distance
        :param x: Vector
        :param y: Another vector
        :return: Distance between x and y
        """
        pass


class Euclidean(Distance):

    def __init__(self, degree):
        """
        Euclidean distance constructor
        :param degree: Degree of Euclidean distance
        """

        super(Euclidean, self).__init__()
        self.degree = degree

    def fit(self, x, y):
        """
        Compute the distance
        :param x: Vector
        :param y: Another vector
        :return: Euclidean distance between x and y
        """
        abs_dist = np.abs(x-y)
        pow_dist = abs_dist ** self.degree
        sum_dist = np.sum(pow_dist, axis=-1)
        dist = sum_dist ** (1/self.degree)

        return dist
