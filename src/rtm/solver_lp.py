import numpy as np

from rtm import algorithms
from rtm import payoffs


def apply_rt(nfg, T):
    """Apply the reward transfer matrix, T, to the matrix game, nfg"""
    it = np.nditer(nfg, flags=['multi_index'])
    prt = np.empty_like(nfg)
    while not it.finished:
        values = np.array(it[0].item(0))
        idx = tuple(it.multi_index)
        # if row player receives reward, it is redistributed to column
        # player i receives the outcome * column i
        prt_values = tuple(np.sum(values * T.transpose(), axis=1))
        prt[idx] = prt_values
        it.iternext()

    return prt


def print_rmt_info(n, nfg, T, T_sums, s_star, g_star):
    if n == 2:
        E = np.ones(T.shape) * (1 - s_star) / (n - 1)
        np.fill_diagonal(E, s_star)
        prt_E = apply_rt(nfg, E)
        prt = apply_rt(nfg, T)
        print('nfg =',
              nfg,
              f'e\' = {s_star}',
              'prt_E =',
              prt_E,
              f's\' = {g_star}',
              'T_sums =',
              T_sums,
              'prt_T =',
              prt,
              sep='\n',
              end='\n\n')
    elif n == 3:
        E = np.ones(T.shape) * (1 - s_star) / (n - 1)
        np.fill_diagonal(E, s_star)
        prt_E = apply_rt(nfg, E)
        prt = apply_rt(nfg, T)
        print('nfg =',
              nfg.transpose(2, 0, 1),
              f'e\' = {s_star}',
              'prt_E =',
              prt_E.transpose(2, 0, 1),
              f's\' = {g_star}',
              'T_sums =',
              T_sums,
              'prt_T =',
              prt.transpose(2, 0, 1),
              sep='\n',
              end='\n\n')
    elif n == 4:
        E = np.ones(T.shape) * (1 - s_star) / (n - 1)
        np.fill_diagonal(E, s_star)
        prt_E = apply_rt(nfg, E)
        prt = apply_rt(nfg, T)
        print('nfg = ',
              nfg.transpose(3, 2, 0, 1),
              f'e\' = {s_star}',
              'prt_E = ',
              prt_E.transpose(3, 2, 0, 1),
              f's\' = {g_star}',
              'T_sums = ',
              T_sums,
              'prt = ',
              prt.transpose(3, 2, 0, 1),
              sep='\n',
              end='\n\n')
        # p1 = down 1 row, p2 = across 1 column, p3 down 1 block, p4 = down 2 blocks
        # p1 = down 2 rows, p2 = down 1 row, p3 down 1 block, p4 = down 2 blocks
    elif n < 12:
        print(f'e\' = {s_star}', f's\' = {g_star}', T_sums, sep='\n')
    else:
        print(f'e\' = {s_star}', f's\' = {g_star}', sep='\n')


def create_T_sums(T):
    """This appends additional rows and columns to the reward transfer matrix,
    holding the sum for that row/column."""
    col_sum = np.sum(T, axis=0)
    row_sum = np.append(np.sum(T, axis=1), 0)
    T_sums = np.append(np.append(T, np.expand_dims(col_sum, axis=0), axis=0),
                       np.expand_dims(row_sum, axis=1),
                       axis=1)
    return T_sums


if __name__ == '__main__':
    np.set_printoptions(linewidth=120)
    np.set_printoptions(formatter={'float_kind': "{:.3f}".format})

    for n in [3, 6]:
        nfg = payoffs.generate_matrix(n, payoffs.Tycoon_nPD)

        s_star = algorithms.find_s_star(nfg)
        T, g_star, _, _ = algorithms.find_T_star(nfg, balance=True)
        T_sums = create_T_sums(T)

        print_rmt_info(n, nfg, T, T_sums, s_star, g_star)

    nfg = payoffs.arbitrary_social_dilemma

    s_star = algorithms.find_s_star(nfg)
    T, g_star, _, _ = algorithms.find_T_star(nfg, balance=True)
    T_sums = create_T_sums(T)

    print_rmt_info(3, nfg, T, T_sums, s_star, g_star)

    n = 5
    nfg = payoffs.generate_matrix(n, payoffs.Functional_form_game)

    s_star = algorithms.find_s_star(nfg)
    T, g_star, _, _ = algorithms.find_T_star(nfg, balance=True)
    T_sums = create_T_sums(T)

    print_rmt_info(n, nfg, T, T_sums, s_star, g_star)

    times = []
    for n in range(8, 18, 1):
        nfg = payoffs.generate_matrix(n, payoffs.Functional_form_game)

        T, g_star, formulation_time, solver_time = algorithms.find_T_star(nfg, balance=False)
        times.append(formulation_time + solver_time)
        print(
            f"n: {n}, num bytes: {nfg.nbytes}, formulation_time: {formulation_time:.2f}, solver_time: {solver_time:.2f}, total_time: {formulation_time+solver_time:.2f}"
        )

    times = np.array(times).round(2)
    np.savetxt("times.txt", times, delimiter=",")
