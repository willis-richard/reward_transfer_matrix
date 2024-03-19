from itertools import product
import time
from typing import Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import linprog, minimize
from scipy.stats import entropy

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


def find_s_star(nfg: NDArray) -> float:
    """
    Return the symmetrical self-interest level of a normal-form social dilemma

    Parameters
    ----------
    nfg : NDArray
        A normal-form social dilemma. For n>1 players, this is an n dimensional
        matrix, with each dimension having a length of 2. Each element of the
        matrix is an array of length n, specifying the payouts to the players.

    Returns
    -------
    float
        The symmetrical self-interest level of the game.
    """
    n = len(nfg.shape)

    # maximise self-interest
    # first variable is the diagonal element, second is the off-diagonals
    cost_function = [-1, 0]

    # rows must sum to one
    A_eq_row = [[1, n - 1]]
    b_eq_row = [1]

    it = np.nditer(nfg, flags=['multi_index'])
    A_ub_reward = []
    while not it.finished:
        # print(f'Iterator: {it[0]} {it.multi_index}', end=' ')
        for idx, p in enumerate(it.multi_index):
            if p == 1:  # the player is cooperating
                # find the current rewards
                C_rewards = np.array(it[0].item(0))

                # find the value from defecting
                new_idx = list(it.multi_index)
                new_idx[idx] = 1 - p
                D_rewards = np.array(tuple(nfg[tuple(new_idx)]))

                diff = C_rewards - D_rewards
                # print(it.multi_index, C_rewards, idx, D_rewards, diff)

                # sum of player rewards * rtm coefficients must be negative
                A_ub_reward.append(
                    [diff[idx],
                     np.sum(diff[:idx]) + np.sum(diff[idx + 1:])])

        it.iternext()

    A_ub_reward = np.vstack(A_ub_reward)
    b_ub_reward = np.zeros(len(A_ub_reward))

    res = linprog(cost_function,
                  A_ub=A_ub_reward,
                  b_ub=b_ub_reward,
                  A_eq=A_eq_row,
                  b_eq=b_eq_row,
                  bounds=[(1.0 / n, 1), (0, (n - 1) / n)])

    assert res.success, res

    return res.x[0]


def find_T_star(nfg: NDArray, equality: bool = True, balance: bool = False) -> Tuple[NDArray, float, float, float]:
    """
    This script finds the symmetrical and general and self-interest levels for n-player matrix games for a several values of n. It first creates all of the
    constraints upon the reward transfer matrix in order to make cooperate the
    dominant action for all players, and uses an optimiser to solve.
    """

    # we assume the variables are of the following form:
    # the reward transfer matrix flattened with the auxiliary variable appended
    # Instead of maximising the minimum of the diagonals, we will minimise
    # the auxiliary variable, z, subject to z being less than the diagonals
    start_time = time.perf_counter()

    n = len(nfg.shape)

    # minimise auxiliary variable z
    cost_function = [0] * n**2 + [-1]

    # require z be less than the diagonal elements
    A_ub_aux = np.array([-np.eye(n**2)[i * (n + 1)] for i in range(n)])
    # append the auxiliary variable
    A_ub_aux = np.c_[A_ub_aux, np.ones(n)]
    b_ub_aux = np.zeros(n)

    # rows must sum to one
    A_row = np.zeros((n, n**2))
    rows, cols = np.divmod(np.arange(n**2), n)
    # Use broadcasting to set the appropriate elements to 1
    A_row[rows, cols + rows * n] = 1
    # append the auxiliary variable
    A_row = np.c_[A_row, [0] * n]
    b_row = np.ones(n)

    it = np.nditer(nfg, flags=['multi_index'])
    A_ub_reward = []
    while not it.finished:
        # print(f'Iterator: {it[0]} {it.multi_index}', end=' ')
        for idx, p in enumerate(it.multi_index):
            if p == 1:  # the player is cooperating
                # find the current rewards
                C_rewards = np.array(it[0].item(0))

                # find the value from defecting
                new_idx = list(it.multi_index)
                new_idx[idx] = 1 - p
                D_rewards = np.array(tuple(nfg[tuple(new_idx)]))

                # print(it.multi_index, C_rewards, idx, D_rewards)

                diff = C_rewards - D_rewards

                # sum of player rewards * rtm coefficients must be negative
                rtm_matrix = np.zeros((n, n))
                rtm_matrix[:, idx] = diff
                A_ub_reward += [np.append(rtm_matrix.flatten(), 0)]

        it.iternext()

    A_ub_reward = np.vstack(A_ub_reward)
    b_ub_reward = np.zeros(len(A_ub_reward))

    if equality:
        A_ub = np.r_[A_ub_aux, A_ub_reward]
        b_ub = np.concatenate((b_ub_aux, b_ub_reward))
        A_eq = A_row
        b_eq = b_row
    else:
        A_ub = np.r_[A_ub_aux, A_ub_reward, A_row]
        b_ub = np.concatenate((b_ub_aux, b_ub_reward, b_row))
        A_eq = None
        b_eq = None


    checkpoint_time = time.perf_counter()

    res = linprog(cost_function,
                  method="highs-ds",
                  A_ub=A_ub,
                  b_ub=b_ub,
                  A_eq=A_eq,
                  b_eq=b_eq,
                  bounds=(0, 1))

    end_time = time.perf_counter()

    assert res.success, res

    g_star, x_orig = res.x[-1], res.x[:n**2]

    T = np.reshape(x_orig, (n, n), 'C')

    if n > 2 and balance:
        if equality:
            T = maximise_entropy(T, A_ub_reward[:, :-1], b_ub_reward,
                                A_row[:, :-1], b_row)
        else:
            T = maximise_entropy(T, np.r_[A_ub_reward[:, :-1], A_row[:, :-1]], np.r_[b_ub_reward, b_row], None, None)

    return T, g_star, checkpoint_time - start_time, end_time - checkpoint_time


def maximise_entropy(T: NDArray, A_ub: NDArray, b_ub: NDArray, A_eq: Optional[NDArray], b_eq: Optional[NDArray]):
    n = T.shape[0]
    diagonals = np.diag(T)
    mask = np.eye(n, dtype=bool)
    non_diagonal_elements = T[~mask].flatten()
    sum_non_diagonals = np.sum(non_diagonal_elements)

    def recombine_diagonals(x):
        recombined = np.zeros_like(T)
        recombined[~mask] = x
        recombined[mask] = diagonals
        return recombined

    # maximise the entropy of non-diagonal elements
    def objective(x):
        # return -np.sum(entropy(recombine_diagonals(x), axis=1))
        # return -entropy(x)
        return -np.sum(entropy(x.reshape(n, n-1), axis=1))

    # Original rtm constraints
    def lp_ub_constraints(x):
        return b_ub - A_ub.dot(recombine_diagonals(x).flatten()) + np.finfo(float).eps

    def lp_eq_constraints(x):
        return b_eq - A_eq.dot(recombine_diagonals(x).flatten())

    constraints = [{
        'type': 'ineq',
        'fun': lp_ub_constraints
    }]

    if A_eq is not None:
        constraints += [{
        'type': 'eq',
        'fun': lp_eq_constraints
    }]

    res = minimize(
        objective,
        non_diagonal_elements,
        method='SLSQP',
        bounds=[(0, 1)] * n * (n-1),
        constraints=constraints,
        jac='3-point',
        options={
            'maxiter': 200,  # defaults 100
            'ftol': 1e-7,  # 1e-6
        }
        )

    assert res.success, res

    T = recombine_diagonals(res.x)

    return T
