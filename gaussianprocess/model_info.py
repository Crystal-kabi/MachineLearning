def get_model_log_marginal_likelihood(model):
    """
    Return the log marginal likelihood of the model
    :param model: Model to be evaluated
    :return:log marginal likelihood value of a model
    """

    try:
        return model.log_marginal_likelihood_value_
    except AttributeError:
        return None