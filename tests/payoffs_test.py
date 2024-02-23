from collections import namedtuple
import numpy as np
from typing import Tuple

from common import DTYPE
from payoffs import generate_matrix, Winner_nPD

Game = namedtuple("Game", ["payoff_D", "payoff_C"])

if __name__ == "__main__":
    # yapf: disable
    A = np.array(
        [[(0, 4), (2, 1)],
        [(3, 0), (1, 1)]],
        np.dtype([(f'p{i}', DTYPE) for i in range(2)]))
    # yapf: enable
    B = generate_matrix(2, Winner_nPD)
    np.testing.assert_array_equal(A, B)

    # yapf: disable
    A = np.array(
        [[[(0, 0, 6), (0, 5, 0)],
          [(1, 4, 0), (3, 1, 0)]],
        [[(0, 4, 1), (2, 1, 1)],
          [(3, 0, 1), (1, 1, 1)]]],
        np.dtype([(f'p{i}', DTYPE) for i in range(3)])
    ).transpose((1, 2, 0))
    # yapf: enable
    B = generate_matrix(3, Winner_nPD)
    np.testing.assert_array_equal(A, B)

    # D = 1, C = 0
    # plus, pot = n + n*n_C
    # if player at index n_D players plays D, they take the pot,
    # otherwise all players share it
    # SW = n + n*n_C + n_D
    def payoff_D(idx: Tuple, pid: int):
        n = len(idx)
        n_D = np.count_nonzero(np.array(idx))
        n_C = n - n_D
        pot = n + n * n_C
        if idx[n_D - 1] == 1:  # someone stole the pot
            return 1 + (pot if n_D == pid else 0)
        else:  # no one stole the pot
            return 1 + pot / n

    def payoff_C(idx: Tuple, pid: int):
        n = len(idx)
        n_D = np.count_nonzero(np.array(idx))
        n_C = n - n_D
        pot = n + n * n_C
        if idx[n_D - 1] == 1:  # someone stole the pot
            return 0
        else:  # no one stole the pot
            return pot / n

    # yapf: disable
    A = np.array(
        [[[(4, 4, 4), (3, 4, 3)],
          [(10, 0, 0), (1, 7, 0)]],
        [[(3, 3, 4), (0, 7, 1)],
          [(3, 2, 3), (1, 1, 4)]]],
        np.dtype([(f'p{i}', DTYPE) for i in range(3)])
    ).transpose((1, 2, 0))
    # yapf: enable
    B = generate_matrix(3, Game(payoff_D, payoff_C))
    np.testing.assert_array_equal(A, B)

    # yapf: disable
    A = np.array(
        [[(3, 3), (2, 3)],
        [(5, 0), (1, 3)]],
        np.dtype([(f'p{i}', DTYPE) for i in range(2)]))
    # yapf: enable
    B = generate_matrix(2, Game(payoff_D, payoff_C))
    np.testing.assert_array_equal(A, B)


# As above, but D does not get 1
def payoff_D(idx: Tuple, pid: int):
    n = len(idx)
    n_D = np.count_nonzero(np.array(idx))
    n_C = n - n_D
    pot = n + n * n_C
    if idx[n_D - 1] == 1:  # someone stole the pot
        return pot if n_D == pid else 0
    else:  # no one stole the pot
        return pot / n


# yapf: disable
A = np.array(
    [[[(4, 4, 4), (3, 3, 3)],
      [(9, 0, 0), (0, 6, 0)]],
     [[(3, 3, 3), (0, 6, 0)],
      [(2, 2, 2), (0, 0, 3)]]],
    np.dtype([(f'p{i}', DTYPE) for i in range(3)])
).transpose((1, 2, 0))
# yapf: enable
B = generate_matrix(3, Game(payoff_D, payoff_C))
np.testing.assert_array_equal(A, B)
