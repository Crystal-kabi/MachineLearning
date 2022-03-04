def read_data(path=None, data=None, feature_col=None, target_col=None, split=False, **kwargs):
    """
    Read data and further process
    :param path: File path of the data
    :param data: Data in pd.DataFrame format
    :param feature_col: Name of the feature column
    :param target_col: Name of the target column
    :param split: When True, split the data
    :param kwargs: Optional arguments for train test split
    :return: Whole data ( and additional information when conditions are satisfied)
    """
    # if file path given, read from path first
    if isinstance(path, str):
        # if file path exists, set data to file
        try:
            data = pd.read_csv(path)
        except FileNotFoundError:
            # if file does not exist, do nothing
            pass

    # if data is None at this point, no data can be read
    if data is None:
        raise ValueError("No data found.")

    # if feature column given, feature is the extracted columns
    if feature_col is not None:
        feature = data[feature_col]
    # if no feature column given, feature is the whole data
    else:
        feature = data

    # if target column given, feature is the extracted columns
    if target_col is not None:
        target = data[target_col]
    # if no target columns given, target is set to None
    else:
        target = None

    # if split but no target exist, only split the feature
    # return feature, target, train feature, test feature
    if split and target is None:
        train_feature, test_feature = train_test_split(feature, **kwargs)

        return data, feature, target, train_feature, test_feature

    # if split and target exists, split both the feature and the target
    # return original data, feature, target, train feature, test feature, train target, test target
    elif split:
        train_feature, test_feature, train_target, test_target = train_test_split(feature, target, **kwargs)

        return data, feature, target, train_feature, test_feature, train_target, test_target

    # if not split and no target exists, return feature (which is also the initial data) only
    elif target is None:
        return feature

    # if not split and target exists, return original data, feature and target
    else:
        return data, feature, target
