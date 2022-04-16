import numpy as np


def pos2index(env_size, position):

    """
    Turn position coordinates to index in the environment in a way consistent in 2 dimensions
    :param env_size: int or 1 dimensional iterable. dimensions of the environment.
    :param position: int or 1 dimensional iterable in the order (x1, x2, x3, ..., xn). position on the environment.
    :return: index of the position
    """

    # use as array for easier computation
    env_size_array = np.asarray(env_size)
    position_array = np.asarray(position)

    # given a one dimensional environment, format the size and position as 1 dimensional arrays
    if len(env_size_array.shape) == 0:
        env_size_array = np.expand_dims(env_size_array, 0)
    if len(position_array.shape) == 0:
        position_array = np.expand_dims(position_array, 0)

    # check if the position is valid in the given environment
    if not np.all(position_array < env_size_array):
        raise ValueError("Position must be within the size componentwise.")

    # create array of length = number of possible positions
    all_index = np.asarray(range(env_size_array.prod()))
    all_index_as_env_shape = all_index.reshape(env_size_array[::-1])
    all_index_as_env_shape = all_index_as_env_shape.transpose()

    # get the index
    index = all_index_as_env_shape[tuple(position)]

    return index


def index2pos(env_size, index):

    # use as array for easier computation
    env_size_array = np.asarray(env_size)

    # given a one dimensional environment, format the size as 1 dimensional arrays
    if len(env_size_array.shape) == 0:
        env_size_array = np.expand_dims(env_size_array, 0)

    # create array of length = number of possible positions
    all_index = np.asarray(range(env_size_array.prod()))
    all_index_as_env_shape = all_index.reshape(env_size_array[::-1])
    all_index_as_env_shape = all_index_as_env_shape.transpose()

    # find the coordinate of the position
    position = np.argwhere(all_index_as_env_shape == index)

    if len(position) != 1:
        raise ValueError("Invalid index")

    # transform the position from array to tuple
    position = tuple(position[0])

    return position
