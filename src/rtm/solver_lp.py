"""
This script finds the symmetrical and general and self-interest levels for a
potential n-player game for a several values of n. It first creates all of the
constraints upon the reward transfer matrix in order to make cooperate the
dominant action for all players, and uses an optimiser to solve.
"""

import fractions
import time
from typing import List, Tuple

import numpy as np
from scipy.optimize import linprog

from rtm.payoffs import generate_matrix
from rtm.entropy import maximise_entropy

def find_s_dash(A):
    # First variable is the diagonal element, second is the off-diagonals
    n = len(A.shape)

    # maximise self-interest
    cost_function = [-1, 0]

    # rows must sum to one
    A_eq_row = [[1, n - 1]]
    b_eq_row = [1]

    it = np.nditer(A, flags=['multi_index'])
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
                D_rewards = np.array(tuple(A[tuple(new_idx)]))

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


def find_rtm(A, balance=False):
    # we assume the variables are of the following form:
    # the reward transfer matrix flattened with the auxiliary variable appended
    # Instead of maximising the minimum of the diagonals, we will minimise
    # the auxiliary variable, z, subject to z being less than the diagonals
    n = len(A.shape)

    # minimise z
    cost_function = [0] * n**2 + [-1]

    # require z be less than the diagonal elements
    A_ub_aux = np.array([-np.eye(n**2)[i * (n + 1)] for i in range(n)])
    # append the auxiliary variable
    A_ub_aux = np.c_[A_ub_aux, [1] * n]
    b_ub_aux = np.zeros(n)

    # rows must sum to one
    A_eq_row = np.zeros((n, n**2))
    rows, cols = np.divmod(np.arange(n**2), n)
    # Use broadcasting to set the appropriate elements to 1
    A_eq_row[rows, cols + rows * n] = 1
    # append the auxiliary variable
    A_eq_row = np.c_[A_eq_row, [0] * n]
    b_eq_row = np.ones(n)

    it = np.nditer(A, flags=['multi_index'])
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
                D_rewards = np.array(tuple(A[tuple(new_idx)]))

                # print(it.multi_index, C_rewards, idx, D_rewards)

                diff = C_rewards - D_rewards

                # sum of player rewards * rtm coefficients must be negative
                rtm_matrix = np.zeros((n, n))
                rtm_matrix[:, idx] = diff
                A_ub_reward += [np.append(rtm_matrix.flatten(), 0)]

        it.iternext()

    A_ub_reward = np.vstack(A_ub_reward)
    b_ub_reward = np.zeros(len(A_ub_reward))

    A_ub = np.r_[A_ub_aux, A_ub_reward]
    b_ub = np.concatenate((b_ub_aux, b_ub_reward))
    # print(cost_function, A_ub, b_ub, sep="\n")

    res = linprog(cost_function,
                  A_ub=A_ub,
                  b_ub=b_ub,
                  A_eq=A_eq_row,
                  b_eq=b_eq_row,
                  bounds=(0, 1))

    assert res.success, res
    # print(A_ub.dot(res.x) - b_ub)

    g_dash, x_orig = res.x[-1], res.x[:n**2]

    G = np.reshape(x_orig, (n, n), 'C')

    # if n > 2 and balance and 0.0 in G:
    if n > 2 and balance:
        G = maximise_entropy(G, A_ub_reward[:, :-1], b_ub_reward, A_eq_row[:, :-1], b_eq_row)

    return G, g_dash


def apply_rt(A, G):
    """Apply the reward transfer matrix, G, to the matrix game, A"""
    it = np.nditer(A, flags=['multi_index'])
    prt = np.empty_like(A)
    while not it.finished:
        values = np.array(it[0].item(0))
        idx = tuple(it.multi_index)
        # if row player receives reward, it is redistributed to column
        # player i receives the outcome * column i
        prt_values = tuple(np.sum(values * G.transpose(), axis=1))
        prt[idx] = prt_values
        it.iternext()

    return prt


def print_rmt_info(n, A, G, G_sums, s_dash, g_dash):
    if n == 2:
        E = np.ones(G.shape) * (1 - s_dash) / (n - 1)
        np.fill_diagonal(E, s_dash)
        prt_E = apply_rt(A, E)
        prt = apply_rt(A, G)
        print('A =',
              A,
              f'e\' = {s_dash}',
              'prt_E =',
              prt_E,
              f's\' = {g_dash}',
              'G_sums =',
              G_sums,
              'prt_G =',
              prt,
              sep='\n',
              end='\n\n')
    elif n == 3:
        E = np.ones(G.shape) * (1 - s_dash) / (n - 1)
        np.fill_diagonal(E, s_dash)
        prt_E = apply_rt(A, E)
        prt = apply_rt(A, G)
        print('A =',
              A.transpose(2, 0, 1),
              f'e\' = {s_dash}',
              'prt_E =',
              prt_E.transpose(2, 0, 1),
              f's\' = {g_dash}',
              'G_sums =',
              G_sums,
              'prt_G =',
              prt.transpose(2, 0, 1),
              sep='\n',
              end='\n\n')
    elif n == 4:
        E = np.ones(G.shape) * (1 - s_dash) / (n - 1)
        np.fill_diagonal(E, s_dash)
        prt_E = apply_rt(A, E)
        prt = apply_rt(A, G)
        print('A = ',
              A.transpose(3, 2, 0, 1),
              f'e\' = {s_dash}',
              'prt_E = ',
              prt_E.transpose(3, 2, 0, 1),
              f's\' = {g_dash}',
              'G_sums = ',
              G_sums,
              'prt = ',
              prt.transpose(3, 2, 0, 1),
              sep='\n',
              end='\n\n')
        # p1 = down 1 row, p2 = across 1 column, p3 down 1 block, p4 = down 2 blocks
        # p1 = down 2 rows, p2 = down 1 row, p3 down 1 block, p4 = down 2 blocks
    elif n < 12:
        print(f'e\' = {s_dash}', f's\' = {g_dash}', G_sums, sep='\n')
    else:
        print(f'e\' = {s_dash}', f's\' = {g_dash}', sep='\n')


def create_G_sums(G):
    col_sum = np.sum(G, axis=0)
    row_sum = np.append(np.sum(G, axis=1), 0)
    G_sums = np.append(np.append(G,
                                  np.expand_dims(col_sum, axis=0),
                                  axis=0),
                        np.expand_dims(row_sum, axis=1),
                        axis=1)
    return G_sums


if __name__ == '__main__':
    np.set_printoptions(linewidth=120)
    np.set_printoptions(formatter={'float_kind': "{:.3f}".format})
    if False:
      np.set_printoptions(formatter={
          'all': lambda x: str(fractions.Fraction(x).limit_denominator(500))
      })

    from rtm.payoffs import DTYPE, Tycoon_nPD, Functional_form_game

    for n in [3, 10]:
        A = generate_matrix(n, Tycoon_nPD)

        s_dash = find_s_dash(A)
        G, g_dash = find_rtm(A, balance=True)
        G_sums = create_G_sums(G)

        print_rmt_info(n, A, G, G_sums, s_dash, g_dash)

    # Arbitrary social dilemma
    # yapf: disable
    A = np.array(
        [[[(9, 6, 7), (2, 9, 7)],
        [(8, 4, 8), (3, 2, 1)]],
        [[(1, 6, 12), (0, 5, 2)],
        [(8, 2, 8), (1, 2, 0)]]],
        np.dtype([(f'p{i}', DTYPE) for i in range(3)])
        ).transpose((1, 2, 0))
    # yapf: enable

    s_dash = find_s_dash(A)
    G, g_dash = find_rtm(A, balance=True)
    G_sums = create_G_sums(G)

    print_rmt_info(3, A, G, G_sums, s_dash, g_dash)

    # from payoffs import Functional_form_game as game


    n = 5
    A = generate_matrix(n, Functional_form_game)

    s_dash = find_s_dash(A)
    G, g_dash = find_rtm(A, balance=True)
    G_sums = create_G_sums(G)

    print_rmt_info(n, A, G, G_sums, s_dash, g_dash)

    times: List[float] = []
    for n in range(8,18,1):
        A = generate_matrix(n, Functional_form_game)

        start_time = time.perf_counter()
        G, g_dash = find_rtm(A, balance=False)
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        times.append(end_time - start_time)
        print(n, A.nbytes, elapsed_time)

    times = np.array(times).round(2)
    np.savetxt("times.txt", times, delimiter=",")
