import itertools
import numpy as np
from sklearn.model_selection import KFold
import datetime


def hyper_gridsearch_cv(model, feature, target, hyperparameter_grid, cv, evaluate_func=None, evaluate_on_model=False, minimize=True, silence=False, show_time=True):

    if not isinstance(feature, np.ndarray):
        feature = np.array(feature)
    if not isinstance(target, np.ndarray):
        target = np.array(target)

    kfold = KFold(n_splits=cv)
    k_fold_index = [i for i in kfold.split(feature, target)]

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
        train_target = target[train_index]
        validate_target = target[test_index]

        train_feature_list.append(train_feature)
        validate_feature_list.append(validate_feature)
        train_target_list.append(train_target)
        validate_target_list.append(validate_target)

    start_time = datetime.datetime.now()
    if not silence:
        print(f"Grid search: 0/{len(all_hyperparameter_product)}", end="\r", flush=True)

    for c in all_hyperparameter_product:

        score_on_fold = []

        hyperparameter_choice = {hyperparameter_name[n]: c[n] for n in range(hyperparameter_num)}
        clf = model(**hyperparameter_choice)

        error = False
        for fold_num in range(len(train_feature_list)):

            train_feature = train_feature_list[fold_num]
            validate_feature = validate_feature_list[fold_num]
            train_target = train_target_list[fold_num]
            validate_target = validate_target_list[fold_num]

            # put hyperparameter choice to fit model
            try:
                clf.fit(train_feature, train_target)
            except:
                error = True
                continue

            # evaluate on validation
            if evaluate_func is None:
                score = clf.score(validate_feature, validate_target)
            elif not evaluate_on_model:
                prediction = clf.predict(validate_feature)
                score = evaluate_func(prediction, validate_target)
            else:
                score = evaluate_func(clf)

            score_on_fold.append(score)

        if error:
            if not silence:
                print(f"{hyperparameter_choice} skipped")
            combo_cnt += 1
            continue

        all_fold_mean_score = np.mean(score_on_fold)
        hyperparameter_score.append(all_fold_mean_score)

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

    if not silence:
        print("")

    if minimize:
        best_hyper_index = np.argmin(hyperparameter_score)
    else:
        best_hyper_index = np.argmax(hyperparameter_score)

    best_hyper = all_hyperparameter_product[best_hyper_index]
    best_score = hyperparameter_score[best_hyper_index]
    best_hyper = {hyperparameter_name[n]: best_hyper[n] for n in range(hyperparameter_num)}

    cv_info = {"best_hyper": best_hyper, "best_score": best_score, "hyperparameter_score": hyperparameter_score}

    return cv_info


def model_evaluation_cv(model, hyperparameter, feature, target, cv, evaluate_func=None, evaluate_on_model=False, silence=False, show_time=True):

    if not isinstance(feature, np.ndarray):
        feature = np.array(feature)
    if not isinstance(target, np.ndarray):
        target = np.array(target)

    kfold = KFold(n_splits=cv)
    k_fold_index = kfold.split(feature, target)

    clf = model(**hyperparameter)

    score_on_fold = []
    fold_counter = 1
    start_time = datetime.datetime.now()
    if not silence:
        print(f"Cross validation: Fold = 0/{cv}", end="\r", flush=True)

    for train_index, test_index in k_fold_index:

        train_feature = feature[train_index]
        validate_feature = feature[test_index]
        train_target = target[train_index]
        validate_target = target[test_index]

        clf.fit(train_feature, train_target)

        # evaluate on validation
        if evaluate_func is None:
            score = clf.score(validate_feature, validate_target)
        elif not evaluate_on_model:
            prediction = clf.predict(validate_feature)
            score = evaluate_func(prediction, validate_target)
        else:
            score = evaluate_func(clf)
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

    return score_on_fold


def nested_cv(model, feature, target, hyperparameter_grid, inner_cv, outer_cv, evaluate_func=None, minimize=True, evaluate_on_model=False):

    kfold = KFold(n_splits=outer_cv)
    k_fold_index = kfold.split(feature, target)

    score_on_fold = []
    for outer_train_index, outer_test_index in k_fold_index:
        outer_train_feature = feature[outer_train_index]
        outer_validate_feature = feature[outer_test_index]
        outer_train_target = target[outer_train_index]
        outer_validate_target = target[outer_test_index]

        grid_search_result = hyper_gridsearch_cv(model=model, feature=outer_train_feature, target=outer_train_target, hyperparameter_grid=hyperparameter_grid, evaluate_func=evaluate_func, cv=inner_cv, evaluate_on_model=evaluate_on_model, minimize=minimize)
        hyperparameter = grid_search_result["best_hyper"]
        clf = model(**hyperparameter)
        clf.fit(outer_train_feature, outer_train_target)

        # evaluate on validation
        if evaluate_func is None:
            score = clf.score(outer_validate_feature, outer_validate_target)
        elif not evaluate_on_model:
            prediction = clf.predict(outer_validate_feature)
            score = evaluate_func(prediction, outer_validate_target)
        else:
            score = evaluate_func(clf)
        score_on_fold.append(score)

    return score_on_fold


if __name__ == '__main__':

    from sklearn.svm import SVC
    from sklearn.datasets import load_iris

    hyperparam_grid = {"kernel": ["rbf"], "C": [1, 10, 100], "gamma": [0.01, 0.1]}
    iris = load_iris()
    X_iris = iris.data
    y_iris = iris.target

    a = hyper_gridsearch_cv(model=SVC, feature=X_iris, target=y_iris, hyperparameter_grid=hyperparam_grid, cv=10)
    b = model_evaluation_cv(model=SVC, hyperparameter=a["best_hyper"], feature=X_iris, target=y_iris, cv=10)
    c = nested_cv(model=SVC, feature=X_iris, target=y_iris, hyperparameter_grid=hyperparam_grid, outer_cv=5, inner_cv=5)