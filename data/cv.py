import itertools
import datetime
import numpy as np


def k_fold_split(k, feature, shuffle=False, random_state=0):
    """
    Split a data into folds.
    :param k: Number of folds
    :param feature: Data to be split
    :param shuffle: When True, the data is shuffled before split, resulting in a shuffled split.
    :param random_state: Seeding for shuffle.
    :return: Generator of tuples (train index, validation index)
    """

    total_data_num = len(feature)
    train_group_len = int(total_data_num/k)
    remainder = total_data_num % k
    val_group_len = train_group_len + remainder

    index_array = range(total_data_num)
    if shuffle:
        np.random.seed(random_state)
        np.random.shuffle(index_array)
    else:
        index_array = np.array(index_array)

    for n in range(k):

        val_pos = slice(n*val_group_len, (n+1)*val_group_len)
        val_index = index_array[val_pos]
        train_index = np.delete(index_array, val_pos)

        yield train_index, val_index


def hyper_gridsearch_cv(model, feature, hyperparameter_grid, cv, target=None, evaluate_func=None, metric_hyper=None, evaluate_on_model=False, minimize=True, silence=False, show_time=True):
    """
    Grid search hyperparameters on a model for best performace.
    :param model: Model class to be fit
    :param feature: Feature used to fit the model
    :param hyperparameter_grid: A dictionary with the hyperparameter name as key and corresponding grid as value
    :param cv: Number of fold for validating the performance on a hyperparameter combination
    :param target: Target for the model to be fitted to. Default is None for unsupervised models.
    :param evaluate_func: Function for evaluating the performance of the model.
    :param metric_hyper: Hyperparameters to be passed into the evaluation function. If the evaluating function does not
    require arguments from hyperparameters. Must be given in dictionary form with hyperparameter
    name as key and its name for the argument as value. Default is None.
    :param evaluate_on_model: When True, evaluate on the model instead of prediction result. Default is True.
    :param minimize: When True, choose the hyperparameters with the least evaluation.
    :param silence: When True, do not print any progress. Default False.
    :param show_time: Only turn on when silence is off. When True, show time taken for progress. Default True.
    :return: Information from cross validation.
    """
    if not isinstance(feature, np.ndarray):
        feature = np.array(feature)
    if not isinstance(target, np.ndarray) and target is not None:
        target = np.array(target)

    if metric_hyper is None:
        metric_hyper = {}

    k_fold_index = k_fold_split(k=cv, feature=feature)

    hyperparameter_num = len(hyperparameter_grid)
    hyperparameter_name = list(hyperparameter_grid.keys())
    all_hyperparameter_product = [i for i in itertools.product(*hyperparameter_grid.values())]

    hyperparameter_score = []
    combo_cnt = 1

    train_feature_list = []
    validate_feature_list = []
    train_target_list = []
    validate_target_list = []
    for train_index, test_index in k_fold_index:

        train_feature = feature[train_index]
        validate_feature = feature[test_index]
        train_feature_list.append(train_feature)
        validate_feature_list.append(validate_feature)

        if target is not None:
            train_target = target[train_index]
            validate_target = target[test_index]
            train_target_list.append(train_target)
            validate_target_list.append(validate_target)

    start_time = datetime.datetime.now()
    if not silence:
        print(f"Grid search: 0/{len(all_hyperparameter_product)}", end="\r", flush=True)

    skipped_hyperparameter = []
    for c in all_hyperparameter_product:

        score_on_fold = []

        hyperparameter_choice = {hyperparameter_name[n]: c[n] for n in range(hyperparameter_num)}

        additional_arg = {}
        for metric_key, hyper_key in metric_hyper.items():
            try:
                additional_arg[metric_key] = hyperparameter_choice[hyper_key]
            except (ValueError, KeyError):
                pass

        clf = model(**hyperparameter_choice)

        error = False
        for fold_num in range(len(train_feature_list)):

            train_feature = train_feature_list[fold_num]
            validate_feature = validate_feature_list[fold_num]

            if target is not None:
                train_target = train_target_list[fold_num]
                validate_target = validate_target_list[fold_num]
                # put hyperparameter choice to fit model
                try:
                    clf.fit(train_feature, train_target)
                except:
                    error = True
                    continue

                # evaluate on validation
                try:
                    if evaluate_func is None:
                        score = clf.score(validate_feature, validate_target, **additional_arg)
                    elif not evaluate_on_model:
                        prediction = clf.predict(validate_feature)
                        score = evaluate_func(prediction, validate_target, **additional_arg)
                    else:
                        score = evaluate_func(clf, **additional_arg)
                except:
                    error = True
                    continue

            else:
                try:
                    train_target = clf.fit_predict(train_feature)
                except:
                    error = True
                    continue

                # evaluate on validation
                try:
                    if evaluate_func is None:
                        score = clf.score(train_feature, train_target, **additional_arg)
                    elif not evaluate_on_model:
                        score = evaluate_func(train_feature, train_target, **additional_arg)
                    else:
                        score = evaluate_func(clf, **additional_arg)
                except:
                    error = True
                    continue

            score_on_fold.append(score)

        end_time = datetime.datetime.now()
        total_time = end_time - start_time
        avg_time = total_time/combo_cnt
        if not silence and show_time:
            print(f"Grid search: {combo_cnt}/{len(all_hyperparameter_product)} | Time taken: {total_time} | {avg_time} / grid | ETA: {avg_time*(len(all_hyperparameter_product)-combo_cnt)}", end="\r", flush=True)
        elif not silence:
            print(f"Grid search: {combo_cnt}/{len(all_hyperparameter_product)}", end="\r", flush=True)
        else:
            pass

        combo_cnt += 1

        if error:
            skipped_hyperparameter.append(hyperparameter_choice)
            hyperparameter_score.append(np.nan)
            continue

        all_fold_mean_score = np.mean(score_on_fold)
        hyperparameter_score.append(all_fold_mean_score)

    skipped_num = len(skipped_hyperparameter)
    if not silence:
        print("")
        skipped_cnt = 1
        while skipped_cnt <= skipped_num:
            s_h = skipped_hyperparameter[skipped_cnt-1]
            print(f"Skipped {skipped_cnt}/{skipped_num} |  {s_h} is skipped")
            skipped_cnt += 1

    if minimize:
        best_hyper_index = np.nanargmin(hyperparameter_score)
    else:
        best_hyper_index = np.nanargmax(hyperparameter_score)

    best_hyper = all_hyperparameter_product[best_hyper_index]
    best_score = hyperparameter_score[best_hyper_index]
    best_hyper = {hyperparameter_name[n]: best_hyper[n] for n in range(hyperparameter_num)}

    cv_info = {"best_hyper": best_hyper, "best_score": best_score, "hyperparameter_score": hyperparameter_score}

    return cv_info


def model_evaluation_cv(model, feature, cv, hyperparameter=None, target=None, evaluate_func=None, metric_hyper=None, evaluate_on_model=False, model_info=None, silence=False, show_time=True):
    """
    Model evaluation through cross validation
    :param model: Model class to be evaluated
    :param feature: Feature used to fit the model
    :param cv: Number of folds for cross validation
    :param hyperparameter: Hyperparameters feed into the model
    :param target: Target for the model to fit to for supervised model. Default is None for unsupervised model.
    :param evaluate_func: Function used to evaluate the model.
    :param metric_hyper: Hyperparameters to be passed into the evaluation function. If the evaluating function does not
    require arguments from hyperparameters. Must be given in dictionary form with hyperparameter
    name as key and its name for the argument as value. Default is None.
    :param evaluate_on_model: When True, evaluate on model instead of prediction. Default is False.
    :param model_info: Model attributes to retrieve during cross validation as a list.
    :param silence: When True, do not print any progress. Default False.
    :param show_time: Only turn on when silence is off. When True, show time taken for progress. Default True.
    :return: When model_info is given, return (list of scores on each fold, list of model attributes on each fold).
    Otherwise, return list of scores on each fold.
    """

    if not isinstance(feature, np.ndarray):
        feature = np.array(feature)
    if not isinstance(target, np.ndarray) and target is not None:
        target = np.array(target)

    if hyperparameter is None:
        hyperparameter = {}

    if metric_hyper is None:
        metric_hyper = {}

    additional_arg = {}
    for metric_key, hyper_key in metric_hyper.items():
        try:
            additional_arg[metric_key] = hyperparameter[hyper_key]
        except (ValueError, KeyError):
            pass

    if model_info is None:
        model_info = []

    k_fold_index = k_fold_split(k=cv, feature=feature)

    clf = model(**hyperparameter)

    train_feature_list = []
    validate_feature_list = []
    train_target_list = []
    validate_target_list = []
    for train_index, test_index in k_fold_index:

        train_feature = feature[train_index]
        validate_feature = feature[test_index]
        train_feature_list.append(train_feature)
        validate_feature_list.append(validate_feature)

        if target is not None:
            train_target = target[train_index]
            validate_target = target[test_index]
            train_target_list.append(train_target)
            validate_target_list.append(validate_target)

    model_info_on_fold = []
    score_on_fold = []
    fold_counter = 1
    start_time = datetime.datetime.now()
    if not silence:
        print(f"Cross validation: Fold = 0/{cv}", end="\r", flush=True)
    for fold_num in range(len(train_feature_list)):

        train_feature = train_feature_list[fold_num]
        validate_feature = validate_feature_list[fold_num]

        attr_list = {}

        if target is not None:
            train_target = train_target_list[fold_num]
            validate_target = validate_target_list[fold_num]
            # put hyperparameter choice to fit model
            try:
                clf.fit(train_feature, train_target)
            except:
                score_on_fold.append(np.nan)
                for attr in model_info:
                    attr_list[attr] = np.nan
                model_info_on_fold.append(attr_list)
                continue

            # evaluate on validation
            if evaluate_func is None:
                score = clf.score(validate_feature, validate_target, **additional_arg)
            elif not evaluate_on_model:
                prediction = clf.predict(validate_feature)
                score = evaluate_func(prediction, validate_target, **additional_arg)
            else:
                score = evaluate_func(clf, **additional_arg)

        else:
            try:
                train_target = clf.fit_predict(train_feature)
            except:
                score_on_fold.append(np.nan)
                for attr in model_info:
                    attr_list[attr] = np.nan
                model_info_on_fold.append(attr_list)
                continue

            # evaluate on validation
            if evaluate_func is None:
                score = clf.score(train_feature, train_target, **additional_arg)
            elif not evaluate_on_model:
                score = evaluate_func(train_feature, train_target, **additional_arg)
            else:
                score = evaluate_func(clf, **additional_arg)

        for attr in model_info:
            try:
                attr_list[attr] = clf.__getattribute__(attr)
            except AttributeError:
                attr_list[attr] = np.nan

        model_info_on_fold.append(attr_list)
        score_on_fold.append(score)

        total_time = datetime.datetime.now() - start_time

        if not silence and show_time:
            print(f"Cross validation: Fold = {fold_counter}/{cv} | Time taken: {total_time} | {total_time / fold_counter} / fold",
                  end="\r", flush=True)
        elif not silence:
            print(f"Cross validation: Fold = {fold_counter}/{cv}", end="\r", flush=True)
        else:
            pass

        fold_counter += 1

    if not silence:
        print("")

    if len(model_info) > 0:
        return score_on_fold, model_info_on_fold
    else:
        return score_on_fold


def nested_cv(model, feature, hyperparameter_grid, inner_cv, outer_cv, target=None, inner_evaluate_func=None, outer_evaluate_func=None, metric_hyper=None, minimize=True, inner_evaluate_on_model=False, outer_evaluate_on_model=False, model_info=None, silence=False, show_time=True):
    """
    Nested cross validation on models to optmise hyperparameters and evaluate performance.
    :param model: Model class to be optimised and evaluated
    :param feature: Feature used to fit the model
    :param hyperparameter_grid: A dictionary with the hyperparameter name as key and corresponding grid as value
    :param inner_cv: Hyperparameter optimisation cross validation fold.
    :param outer_cv: Number of fold for validating the performance on a hyperparameter combination
    :param target:  Target for the model to be fitted to. Default is None for unsupervised models.
    :param inner_evaluate_func: Function for evaluating the performance of the model on choosing hyperparameters.
    :param outer_evaluate_func: Function for evaluating the performance of the model for model comparison.
    :param metric_hyper: Hyperparameters to be passed into the evaluation function. If the evaluating function does not
    require arguments from hyperparameters. Must be given in dictionary form with hyperparameter
    name as key and its name for the argument as value. Default is None.
    :param minimize: When True, choose the hyperparameters with the least evaluation.
    :param inner_evaluate_on_model: When True, choose hyperparamters based on the model instead of prediction result. Default is True.
    :param outer_evaluate_on_model: When True, evaluate on the model instead of prediction result. Default is True.
    :param model_info: Model attributes to retrieve during cross validation as a list.
    :param silence: When True, do not print any progress. Default False.
    :param show_time: Only turn on when silence is off. When True, show time taken for progress. Default True.
    :return: When model_info is given, return (Grid search result, list of scores on each fold, list of model attributes on each fold).
    Otherwise, return (Grid search result, list of scores on each fold).
    """

    if not isinstance(feature, np.ndarray):
        feature = np.array(feature)
    if not isinstance(target, np.ndarray) and target is not None:
        target = np.array(target)

    if outer_evaluate_func is None:
        outer_evaluate_func = inner_evaluate_func

    if model_info is None:
        model_info = []
    if isinstance(model_info, str):
        model_info = [model_info]

    k_fold_index = k_fold_split(k=outer_cv, feature=feature)

    if not silence:
        print(f"Cross validation: Fold = 0/{outer_cv}", end="\r", flush=True)

    outer_train_feature_list = []
    outer_validate_feature_list = []
    outer_train_target_list = []
    outer_validate_target_list = []
    for train_index, test_index in k_fold_index:

        outer_train_feature = feature[train_index]
        outer_validate_feature = feature[test_index]
        outer_train_feature_list.append(outer_train_feature)
        outer_validate_feature_list.append(outer_validate_feature)

        if target is not None:
            outer_train_target = target[train_index]
            outer_validate_target = target[test_index]
            outer_train_target_list.append(outer_train_target)
            outer_validate_target_list.append(outer_validate_target)

    grid_search_result_list = []
    model_info_on_fold = []
    score_on_fold = []
    fold_counter = 1
    start_time = datetime.datetime.now()
    for fold_num in range(len(outer_train_feature_list)):

        outer_train_feature = outer_train_feature_list[fold_num]
        outer_validate_feature = outer_validate_feature_list[fold_num]

        attr_list = {}

        if target is not None:
            outer_train_target = outer_train_target_list[fold_num]
            outer_validate_target = outer_validate_target_list[fold_num]
        else:
            outer_train_target = None
            outer_validate_target = None

        grid_search_result = hyper_gridsearch_cv(model=model, feature=outer_train_feature, target=outer_train_target, hyperparameter_grid=hyperparameter_grid, evaluate_func=inner_evaluate_func, metric_hyper=metric_hyper, cv=inner_cv, evaluate_on_model=inner_evaluate_on_model, minimize=minimize, silence=silence, show_time=show_time)
        grid_search_result_list.append(grid_search_result)
        hyperparameter = grid_search_result["best_hyper"]

        if metric_hyper is None:
            metric_hyper = {}
        additional_arg = {}
        for metric_key, hyper_key in metric_hyper.items():
            try:
                additional_arg[metric_key] = hyperparameter[hyper_key]
            except (ValueError, KeyError):
                pass

        clf = model(**hyperparameter)

        if target is not None:

            # put hyperparameter choice to fit model
            try:
                clf.fit(outer_train_feature, outer_train_target)
            except:
                score_on_fold.append(np.nan)
                for attr in model_info:
                    attr_list[attr] = np.nan
                model_info_on_fold.append(attr_list)
                continue

            # evaluate on validation
            if outer_evaluate_func is None:
                score = clf.score(outer_validate_feature, outer_validate_target, **additional_arg)
            elif not outer_evaluate_on_model:
                prediction = clf.predict(outer_validate_feature)
                score = outer_evaluate_func(prediction, outer_validate_target, **additional_arg)
            else:
                score = outer_evaluate_func(clf)

        else:
            try:
                outer_train_target = clf.fit_predict(outer_train_feature, **additional_arg)
            except:
                score_on_fold.append(np.nan)
                for attr in model_info:
                    attr_list[attr] = np.nan
                model_info_on_fold.append(attr_list)
                continue

            # evaluate on validation
            if outer_evaluate_func is None:
                score = clf.score(outer_train_feature, outer_train_target, **additional_arg)
            elif not outer_evaluate_func:
                score = outer_evaluate_func(outer_train_feature, outer_train_target, **additional_arg)
            else:
                score = outer_evaluate_func(clf, **additional_arg)

        for attr in model_info:
            try:
                attr_list[attr] = clf.__getattribute__(attr)
            except AttributeError:
                attr_list[attr] = np.nan

        model_info_on_fold.append(attr_list)
        score_on_fold.append(score)

        total_time = datetime.datetime.now() - start_time

        if not silence and show_time:
            print(f"Cross validation: Fold = {fold_counter}/{outer_cv} | Time taken: {total_time} | {total_time / fold_counter} / fold",
                  end="\r", flush=True)
        elif not silence:
            print(f"Cross validation: Fold = {fold_counter}/{outer_cv}", end="\r", flush=True)
        else:
            pass

        fold_counter += 1

    if not silence:
        print("")

    if len(model_info) > 0:
        return grid_search_result_list, score_on_fold, model_info_on_fold

    return grid_search_result_list, score_on_fold


if __name__ == '__main__':

    from sklearn.svm import SVC
    from sklearn.datasets import load_iris

    hyperparam_grid = {"kernel": ["rbf"], "C": [1, 10, 100], "gamma": [0.01, 0.1]}
    iris = load_iris()
    X_iris = iris.data
    y_iris = iris.target

    a = hyper_gridsearch_cv(model=SVC, feature=X_iris, target=y_iris, hyperparameter_grid=hyperparam_grid, cv=10)
    b = model_evaluation_cv(model=SVC, hyperparameter=a["best_hyper"], feature=X_iris, target=y_iris, cv=10)
    c = nested_cv(model=SVC, feature=X_iris, target=y_iris, hyperparameter_grid=hyperparam_grid, outer_cv=5, inner_cv=5, model_info=["coef0", "gamma"])
