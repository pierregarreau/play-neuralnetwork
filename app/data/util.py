import numpy as np


def transform_integer_class_to_binary_vector(labels: np.array) -> np.array:
    max_label = labels.max()
    identity = np.eye(max_label, max_label)
    return np.array(map(lambda x: identity[x - 1, :], labels))


def not_x_and_not_y(x: int, y: int) -> int:
    if ((x == 0) & (y == 0)):
        return 1
    else:
        return 0


def xnor(x: int, y: int) -> int:
    if ((x == 1) & (y == 0)):
        return 0
    elif ((y == 1) & (x == 0)):
        return 0
    else:
        return 1


def nor(x: int, y: int) -> int:
    if ((x == 1) & (y == 0)):
        return 1
    elif ((y == 1) & (x == 0)):
        return 1
    else:
        return 0
