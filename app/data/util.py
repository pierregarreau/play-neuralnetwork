import numpy as np


def binary_vector_to_class(targets):
        # TODO rename
        if targets.shape.__len__() > 1:
            if targets.shape[1] > 1:
                return np.array(list(map(lambda x: x.argmax() + 1, targets)))
            else:
                return np.array(list(map(lambda x: np.floor(x + 0.5), targets)))
        else:
            return np.array(list(map(lambda x: np.floor(x + 0.5), targets)))


def integer_class_to_binary_vector(labels: np.array) -> np.array:
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
