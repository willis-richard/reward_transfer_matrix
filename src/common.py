from itertools import product
import numpy as np
from typing import Tuple

DTYPE = np.float64


def apply_rt(A, G):
    # if row player receives reward, it is redistributed to column
    # player i receives the outcome * column i
    it = np.nditer(A, flags=['multi_index'])
    prt = np.empty_like(A)
    while not it.finished:
        values = np.array(it[0].item(0))
        idx = tuple(it.multi_index)
        prt_values = tuple(np.sum(values * G.transpose(), axis=1))
        prt[idx] = prt_values
        it.iternext()

    return prt


def generate_combinations(n, fixed_position, action):
    if fixed_position >= n or fixed_position < 0:
        raise ValueError(
            "The fixed position must be within the range of number of players."
        )

    combinations = product([0, 1], repeat=n - 1)

    for combination in combinations:
        combination = list(combination)
        combination.insert(fixed_position, action)
        yield combination
