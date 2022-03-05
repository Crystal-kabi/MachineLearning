import itertools
from sklearn.gaussian_process.kernels import RBF
from sklearn.metrics.pairwise import polynomial_kernel, sigmoid_kernel


def create_kernel_grid(kernel, kernel_hyper_grid, scale_factor_grid=None):

    """
    Create a grid of kernels
    :param kernel: A non-composite kernel object
    :param kernel_hyper_grid: The hyperparameter grids to be supplied to the kernel
    :param scale_factor_grid: The scale factor grid to be multified to the kernel
    :return: Grid of kernels
    """

    if scale_factor_grid is None:
        scale_factor_grid = [1]

    kernel_hyper_num = len(kernel_hyper_grid)
    kernel_hyper_name = list(kernel_hyper_grid.keys())
    kernel_grid_value = [i for i in itertools.product(*kernel_hyper_grid.values())]
    kernel_grid_kwarg = [{kernel_hyper_name[n]: c[n] for n in range(kernel_hyper_num)} for c in kernel_grid_value]
    kernel_grid = [kernel(**kwargs) for kwargs in kernel_grid_kwarg]
    scaled_kernel_grid = []
    for k in kernel_grid:
        error = False
        for i in scale_factor_grid:
            try:
                scaled_kernel_grid.append(i*k)
            except TypeError:
                    error = True
        if error:
            scaled_kernel_grid.append(k)

    return scaled_kernel_grid


class RBFKernel:

    def __init__(self, length_scale=1.0, length_scale_bound=(1e-5, 1e5)):

        self.length_scale = length_scale
        self.length_scale_bound = length_scale_bound

        self.X = None
        self.Y = None

    def __repr__(self):

        rep = f"RBFKernel(length_scale={self.length_scale})"

        return rep

    def fit_transform(self, X, Y=None):

        rbf = RBF(length_scale=self.length_scale, length_scale_bounds=self.length_scale_bound)

        self.X = X
        self.Y = Y

        X = rbf.__call__(X=X, Y=Y)

        return X

    def fit(self, X, Y=None):

        self.fit_transform(X=X, Y=Y)

    def transform(self, X, Y=None):

        rbf = RBF(length_scale=self.length_scale, length_scale_bounds=self.length_scale_bound)

        X = rbf.__call__(X=X, Y=Y)

        return X


class PolyKernel:

    def __init__(self, degree=3, gamma=None, coef0=1):

        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0

        self.X = None
        self.Y = None

    def __repr__(self):

        rep = f"PolynomialKernel(degree={self.degree}, gamma={self.gamma}, coef0={self.coef0})"

        return rep

    def fit_transform(self, X, Y=None):

        self.X = X
        self.Y = Y

        X = polynomial_kernel(X=X, Y=Y, degree=self.degree, gamma=self.gamma, coef0=self.coef0)

        return X

    def fit(self, X, Y=None):

        self.fit_transform(X=X, Y=Y)

    def transform(self, X, Y=None):

        X = polynomial_kernel(X=X, Y=Y, degree=self.degree, gamma=self.gamma, coef0=self.coef0)

        return X


class SigmoidKernel:

    def __init__(self, gamma=None, coef0=1):

        self.gamma = gamma
        self.coef0 = coef0

        self.X = None
        self.Y = None

    def __repr__(self):

        rep = f"SigmoidKernel(gamma={self.gamma}, coef0={self.coef0})"

        return rep

    def fit_transform(self, X, Y=None):

        self.X = X
        self.Y = Y

        X = sigmoid_kernel(X=X, Y=Y, gamma=self.gamma, coef0=self.coef0)

        return X

    def fit(self, X, Y=None):

        self.fit_transform(X=X, Y=Y)

    def transform(self, X, Y=None):

        X = sigmoid_kernel(X=X, Y=Y, gamma=self.gamma, coef0=self.coef0)

        return X

