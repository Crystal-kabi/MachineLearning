import numpy as np


def standard_up(position):

    position = np.asarray(position)

    position[1] += 1

    position = tuple(position)

    return position


def standard_down(position):

    position = np.asarray(position)

    position[1] -= 1

    position = tuple(position)

    return position


def standard_left(position):

    position = np.asarray(position)

    position[0] -= 1

    position = tuple(position)

    return position


def standard_right(position):

    position = np.asarray(position)

    position[0] += 1

    position = tuple(position)

    return position


def standard_rest(position):

    return position


standard_action = {0: standard_rest, 1: standard_up, 2: standard_down, 3: standard_right, 4: standard_left}
standard_size = (8, 8)
standard_penalty = -10
