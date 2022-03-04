from sklearn.gaussian_process import GaussianProcessRegressor

import itertools

import joblib
import json


def regressor(feature, target, test=False, test_feature=None, test_target=None, silence=False, model_save=None,
              info_save=None, **kwargs):
    """
    Fit a Gaussian Process Regressor
    :param feature: Features to be fitted with
    :param target: Targets to be fitted to
    :param test: When True, test the model
    :param test_feature: Features to be tested with
    :param test_target: Targets of the test features
    :param silence: When True, do not print any information on the model
    :param model_save: When given, the model is saved into this value
    :param info_save: When given, information regarding the model is saved into this value
    :param kwargs: Optional keyword arguments for GaussianProcessRegressor
    :return: model, information of the model
    """
    # save the model information
    model_info = dict()

    # fit a gaussian process regressor using provided kernel and arguments
    gp = GaussianProcessRegressor(**kwargs)
    gp.fit(feature, target)

    # compute log marginal likelihood of the data
    # if silence is set to False, print the log marginal likelihood value of the data for this model
    log_likelihood = gp.log_marginal_likelihood_value_
    model_info["log_marginal_likelihood_value"] = log_likelihood
    if not silence:
        print(f"log_marginal_likelihood_value_: {log_likelihood}")

    # if test is set to True, test on the test feature and test target
    # if test feature or test target (or both) is not given, do not test the model
    if test_feature is None or test_target is None:
        test = False
    if test:
        test_score = gp.score(test_feature, test_target)
        model_info["test_score"] = test_score
        # if silence is set to False, print the test score
        if not silence:
            print(f"test score: {test_score}")

    # if model_save is a string, try to save the fitted model
    # if not a legitimate path, do nothing, model will not be saved
    if isinstance(model_save, str):
        try:
            joblib.dump(gp, model_save)
        except FileNotFoundError:
            pass
    # if model_save is a string, try to save the model evaluation information
    # if not a legitimate path, do nothing, information will not be saved
    if isinstance(info_save, str):
        try:
            json.dump(model_info, open(info_save, "w"))
        except FileNotFoundError:
            pass

    return gp, model_info


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
        for i in scale_factor_grid:
            scaled_kernel_grid.append(i*k)

    return scaled_kernel_grid
