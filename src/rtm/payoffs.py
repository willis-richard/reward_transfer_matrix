"""This file defines the payoffs for the games used in the paper, namely
the Symmetrical, Cyclical, Tycoon and Circular variants, each for either
Prisoner's Dilemma, Chicken or Stag Hunt base games."""


import math
from itertools import product

import numpy as np

DTYPE = np.float64


class BaseGame:

    @classmethod
    def payoff_D(cls, idx: tuple[int, ...], pid: int, **kwargs):
        pass

    @classmethod
    def payoff_C(cls, idx: tuple[int, ...], pid: int, **kwargs):
        pass


def payoff(game: BaseGame, idx: tuple[int, ...], pid: int, **kwargs):
    if idx[pid - 1] == 0:
        return game.payoff_C(idx, pid, **kwargs)
    if idx[pid - 1] == 1:
        return game.payoff_D(idx, pid, **kwargs)
    assert False, f"Invalid action {idx[pid-1]} taken"


def generate_matrix(n: int, game: BaseGame, **kwargs):
    M = np.empty([2] * n, dtype=np.dtype([(f'p{i}', DTYPE) for i in range(n)]))
    it = np.nditer(M, flags=['multi_index'])
    while not it.finished:
        idx = tuple(it.multi_index)
        rewards = tuple(payoff(game, idx, i + 1, **kwargs) for i in range(n))
        M[idx] = rewards
        it.iternext()

    return M


def print_game_by_action_profile(n, game):
    for c in product([0, 1], repeat=n):
        print(c, np.array([payoff(game, c, i + 1) for i in range(n)]))


class Symmetrical_nPD(BaseGame):

    @classmethod
    def payoff_D(cls, idx: tuple[int, ...], pid: int, d=1, c=3):
        n = len(idx)
        n_D = np.count_nonzero(np.array(idx))
        n_C = n - n_D
        return d + c * n_C / (n - 1)

    @classmethod
    def payoff_C(cls, idx: tuple[int, ...], pid: int, d=1, c=3):
        n = len(idx)
        n_D = np.count_nonzero(np.array(idx))
        n_C = n - n_D
        return c * (n_C - 1) / (n - 1)


class Symmetrical_nCH(BaseGame):

    @classmethod
    def payoff_D(cls, idx: tuple[int, ...], pid: int, d=1, c=3):
        n = len(idx)
        n_D = np.count_nonzero(np.array(idx))
        n_C = n - n_D
        return (d + c) * n_C / (n - 1)

    @classmethod
    def payoff_C(cls, idx: tuple[int, ...], pid: int, d=1, c=3):
        n = len(idx)
        n_D = np.count_nonzero(np.array(idx))
        n_C = n - n_D
        return c * (n_C - 1) / (n - 1) + d * n_D / (n - 1)


class Symmetrical_nSH(BaseGame):

    @classmethod
    def payoff_D(cls, idx: tuple[int, ...], pid: int, d=1, c=3):
        n = len(idx)
        n_D = np.count_nonzero(np.array(idx))
        n_C = n - n_D
        return c * n_C / (n - 1) + d * (n_D - 1) / (n - 1)

    @classmethod
    def payoff_C(cls, idx: tuple[int, ...], pid: int, d=1, c=3):
        n = len(idx)
        n_D = np.count_nonzero(np.array(idx))
        n_C = n - n_D
        return (c + d) * (n_C - 1) / (n - 1)


class Cyclical_nPD(BaseGame):
    @classmethod
    def payoff_D(cls, idx: tuple[int, ...], pid: int, d=1, c=3):
        n = len(idx)
        if idx[pid % n] == 0:
            return d + c
        else:
            return d

    @classmethod
    def payoff_C(cls, idx: tuple[int, ...], pid: int, d=1, c=3):
        n = len(idx)
        if idx[pid % n] == 0:
            return c
        else:
            return 0


class Cyclical_nCH(BaseGame):

    @classmethod
    def payoff_D(cls, idx: tuple[int, ...], pid: int, d=1, c=3):
        n = len(idx)
        if idx[pid % n] == 0:
            return d + c
        else:
            return 0

    @classmethod
    def payoff_C(cls, idx: tuple[int, ...], pid: int, d=1, c=3):
        n = len(idx)
        if idx[pid % n] == 0:
            return c
        else:
            return d


class Cyclical_nSH(BaseGame):

    @classmethod
    def payoff_D(cls, idx: tuple[int, ...], pid: int, d=1, c=3, k=0):
        n = len(idx)
        if idx[pid % n] == 0:
            return k + c
        else:
            return k + 1

    @classmethod
    def payoff_C(cls, idx: tuple[int, ...], pid: int, d=1, c=3, k=0):
        n = len(idx)
        if idx[pid % n] == 0:
            return k + c + d
        else:
            return k + 0


class Tycoon_nPD(BaseGame):

    @classmethod
    def payoff_D(cls, idx: tuple[int, ...], pid: int, d=1, c=3, k=0):
        n = len(idx)
        n_D = np.count_nonzero(np.array(idx))
        n_C = n - n_D
        if pid == 1:
            return k + d * (n - 1) + c * n_C
        else:
            return k + d + (c if idx[0] == 0 else 0)

    @classmethod
    def payoff_C(cls, idx: tuple[int, ...], pid: int, d=1, c=3, k=0):
        n = len(idx)
        n_D = np.count_nonzero(np.array(idx))
        n_C = n - n_D
        if pid == 1:
            return k + c * (n_C - 1)
        else:
            return k + (c if idx[0] == 0 else 0)


class Tycoon_nCH(BaseGame):

    @classmethod
    def payoff_D(cls, idx: tuple[int, ...], pid: int, d=1, c=3, k=0):
        n = len(idx)
        n_D = np.count_nonzero(np.array(idx))
        n_C = n - n_D
        if pid == 1:
            return k + (c + d) * n_C
        else:
            return k + ((c + d) if idx[0] == 0 else 0)

    @classmethod
    def payoff_C(cls, idx: tuple[int, ...], pid: int, d=1, c=3, k=0):
        n = len(idx)
        n_D = np.count_nonzero(np.array(idx))
        n_C = n - n_D
        if pid == 1:
            return k + c * (n_C - 1) + d * n_D
        else:
            return k + (c if idx[0] == 0 else d)


class Tycoon_nSH(BaseGame):

    @classmethod
    def payoff_D(cls, idx: tuple[int, ...], pid: int, d=1, c=3):
        n = len(idx)
        n_D = np.count_nonzero(np.array(idx))
        n_C = n - n_D
        if pid == 1:
            return c * n_C + d * (n_D - 1)
        else:
            return c if idx[0] == 0 else d

    @classmethod
    def payoff_C(cls, idx: tuple[int, ...], pid: int, d=1, c=3):
        n = len(idx)
        n_D = np.count_nonzero(np.array(idx))
        n_C = n - n_D
        if pid == 1:
            return (c + d) * (n_C - 1)
        else:
            return c + d if idx[0] == 0 else 0


class Circular_nPD(BaseGame):

    @classmethod
    def payoff_D(cls, idx: tuple[int, ...], pid: int, d=1, c=3):
        n = len(idx)
        n_D = np.count_nonzero(np.array(idx))
        n_C = n - n_D
        pids = np.arange(1, n + 1, 1)
        distance = np.array(
            [min(abs(pid - p), n - abs(pid - p)) for p in pids])

        distance[pid - 1] = 1
        weight = (1.0 / 2)**distance
        weight[pid - 1] = 0

        r_v_c = d + c
        r_v_d = d
        values = np.array(idx)
        c_idx = values == 0
        d_idx = values == 1
        values[c_idx] = r_v_c
        values[d_idx] = r_v_d
        return np.dot(values, weight)

    @classmethod
    def payoff_C(cls, idx: tuple[int, ...], pid: int, d=1, c=3):
        n = len(idx)
        n_D = np.count_nonzero(np.array(idx))
        n_C = n - n_D
        pids = np.arange(1, n + 1, 1)
        distance = np.array(
            [min(abs(pid - p), n - abs(pid - p)) for p in pids])

        distance[pid - 1] = 1
        weight = (1.0 / 2)**distance
        weight[pid - 1] = 0

        r_v_c = c
        r_v_d = 0
        values = np.array(idx)
        c_idx = values == 0
        d_idx = values == 1
        values[c_idx] = r_v_c
        values[d_idx] = r_v_d
        return np.dot(values, weight)


class Circular_nCH(BaseGame):

    @classmethod
    def payoff_D(cls, idx: tuple[int, ...], pid: int, d=1, c=3):
        n = len(idx)
        n_D = np.count_nonzero(np.array(idx))
        n_C = n - n_D
        pids = np.arange(1, n + 1, 1)
        distance = np.array(
            [min(abs(pid - p), n - abs(pid - p)) for p in pids])

        distance[pid - 1] = 1
        weight = (1.0 / 2)**distance
        weight[pid - 1] = 0

        r_v_c = d + c
        r_v_d = 0
        values = np.array(idx)
        c_idx = values == 0
        d_idx = values == 1
        values[c_idx] = r_v_c
        values[d_idx] = r_v_d
        return np.dot(values, weight)

    @classmethod
    def payoff_C(cls, idx: tuple[int, ...], pid: int, d=1, c=3):
        n = len(idx)
        n_D = np.count_nonzero(np.array(idx))
        n_C = n - n_D
        pids = np.arange(1, n + 1, 1)
        distance = np.array(
            [min(abs(pid - p), n - abs(pid - p)) for p in pids])

        distance[pid - 1] = 1
        weight = (1.0 / 2)**distance
        weight[pid - 1] = 0

        r_v_c = c
        r_v_d = d
        values = np.array(idx)
        c_idx = values == 0
        d_idx = values == 1
        values[c_idx] = r_v_c
        values[d_idx] = r_v_d
        return np.dot(values, weight)


class Circular_nSH(BaseGame):

    @classmethod
    def payoff_D(cls, idx: tuple[int, ...], pid: int, d=1, c=3):
        n = len(idx)
        n_D = np.count_nonzero(np.array(idx))
        n_C = n - n_D
        pids = np.arange(1, n + 1, 1)
        distance = np.array(
            [min(abs(pid - p), n - abs(pid - p)) for p in pids])

        distance[pid - 1] = 1
        weight = (1.0 / 2)**distance
        weight[pid - 1] = 0

        r_v_c = c
        r_v_d = d
        values = np.array(idx)
        c_idx = values == 0
        d_idx = values == 1
        values[c_idx] = r_v_c
        values[d_idx] = r_v_d
        return np.dot(values, weight)

    @classmethod
    def payoff_C(cls, idx: tuple[int, ...], pid: int, d=1, c=3):
        n = len(idx)
        n_D = np.count_nonzero(np.array(idx))
        n_C = n - n_D
        pids = np.arange(1, n + 1, 1)
        distance = np.array(
            [min(abs(pid - p), n - abs(pid - p)) for p in pids])

        distance[pid - 1] = 1
        weight = (1.0 / 2)**distance
        weight[pid - 1] = 0

        r_v_c = c + d
        r_v_d = 0
        values = np.array(idx)
        c_idx = values == 0
        d_idx = values == 1
        values[c_idx] = r_v_c
        values[d_idx] = r_v_d
        return np.dot(values, weight)


# Arbitrary social dilemma
# yapf: disable
arbitrary_social_dilemma = np.array(
    [[[(9, 6, 7), (2, 9, 7)],
      [(8, 4, 8), (3, 2, 1)]],
     [[(1, 6, 12), (0, 5, 2)],
      [(8, 2, 8), (1, 2, 0)]]],
    np.dtype([(f'p{i}', DTYPE) for i in range(3)])
    ).transpose((1, 2, 0))
# yapf: enable


# yapf: disable
too_many_cooks_in_prison = np.array(
    [[[(2, 2, 2), (3/2, 4, 3/2)],
    [(4, 3/2, 3/2), (5/2, 5/2, 0)]],
    [[(3/2, 3/2, 4), (0, 5/2, 5/2)],
    [(5/2, 0, 5/2), (0, 0, 0)]]],
    np.dtype([(f'p{i}', float) for i in range(3)])
).transpose((1, 2, 0))
# yapf: enable


class Functional_form_game:
    # d is the relative weight of the social welfare to a defecting agent vs cooperating
    @classmethod
    def payoff_D(cls, idx: tuple[int, ...], pid: int, d=2, c=3):
        n = len(idx)
        n_D = np.count_nonzero(np.array(idx))
        n_C = n - n_D

        sw = -c / n * n_C**2 + 2 * c * n_C
        total_weight = np.sum((d * np.array(idx) + 1) * np.arange(1, n + 1))

        return d * pid * sw / total_weight

    @classmethod
    def payoff_C(cls, idx: tuple[int, ...], pid: int, d=2, c=3):
        n = len(idx)
        n_D = np.count_nonzero(np.array(idx))
        n_C = n - n_D

        sw = -c / n * n_C**2 + 2 * c * n_C
        total_weight = np.sum((d * np.array(idx) + 1) * np.arange(1, n + 1))

        return pid * sw / total_weight


# game that shows you need a constraint even when a player does not (currently)
# benefit from defecting
# yapf: disable
nfg = np.array(
    [[[(3, 3, 3), (3, 4, 0)],
      [(4, 0, 3), (4, 1, 0)]],
     [[(0, 3, 4), (0, 4, 1)],
      [(0, 4, 3), (1, 1, 1)]]],
    np.dtype([(f'p{i}', DTYPE) for i in range(3)])
).transpose((1, 2, 0))
# yapf: enable


class Scaled_nPD(BaseGame):
    # returns scaled by pid
    @classmethod
    def payoff_D(cls, idx: tuple[int, ...], pid: int, d=1, c=3):
        n = len(idx)
        n_D = np.count_nonzero(np.array(idx))
        n_C = n - n_D
        return pid * (d + c * n_C / (n - 1))

    @classmethod
    def payoff_C(cls, idx: tuple[int, ...], pid: int, d=1, c=3):
        n = len(idx)
        n_D = np.count_nonzero(np.array(idx))
        n_C = n - n_D
        return pid * c * (n_C - 1) / (n - 1)


class PublicGoodsGame(BaseGame):

    @classmethod
    def payoff_D(cls, idx: tuple[int, ...], pid: int, k=2):
        n = len(idx)
        n_D = np.count_nonzero(np.array(idx))
        n_C = n - n_D
        return 1 + k * n_C / n

    @classmethod
    def payoff_C(cls, idx: tuple[int, ...], pid: int, k=2):
        n = len(idx)
        n_D = np.count_nonzero(np.array(idx))
        n_C = n - n_D
        return k * n_C / n
