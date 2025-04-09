import argparse
import importlib
import numpy as np

from rtm import algorithms


def apply_rt(nfg, T):
    """Apply the reward transfer matrix, T, to the matrix game, nfg"""
    it = np.nditer(nfg, flags=["multi_index"])
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


def print_rtm_info(n, nfg, T, s_star, g_star):
    def transpose_for_printing(matrix, n):
        if n == 2:
            return matrix
        elif n == 3:
            return matrix.transpose(2, 0, 1)
        elif n == 4:
            return matrix.transpose(3, 2, 0, 1)
        else:
            assert False

    def create_T_sums(T):
        """This appends additional rows and columns to the reward transfer matrix,
        holding the sum for that row/column."""
        col_sum = np.sum(T, axis=0)
        row_sum = np.append(np.sum(T, axis=1), 0)
        T_sums = np.append(np.append(T, np.expand_dims(col_sum, axis=0), axis=0),
                        np.expand_dims(row_sum, axis=1),
                        axis=1)
        return T_sums

    if n < 5:
        S = np.ones(T.shape) * (1 - s_star) / (n - 1)
        np.fill_diagonal(S, s_star)
        prt_E = apply_rt(nfg, S)
        prt_T = apply_rt(nfg, T)
        print("Normal-form social dilemma:",
              transpose_for_printing(nfg, n),
              f"Symmetrical self-interest level, s^* = {s_star:.3f}",
              "Transformed game under reward exchange given by s^*:",
              transpose_for_printing(prt_E, n),
              f"General self-interest level, g^* = {g_star:.3f}",
              "Minimal reward transfer matrix, T^*, with column and row totals:",
              create_T_sums(T),
              "Transformed game under reward transfer given by T^*:",
              transpose_for_printing(prt_T, n),
              sep="\n\n",
              end="\n\n\n")
    elif n < 12:
        print(f"Symmetrical self-interest level, s^* = {s_star:.3f}",
              f"General self-interest level, g^* = {g_star:.3f}",
              "Minimal reward transfer matrix, T^*, with column and row totals:",
              create_T_sums(T),
              sep="\n\n",
              end="\n\n\n")
    else:
        print(f"Symmetrical self-interest level, s^* = {s_star:.3f}",
              f"General self-interest level, g^* = {g_star:.3f}",
              sep="\n",
              end="\n\n")


if __name__ == "__main__":
    np.set_printoptions(linewidth=120)
    np.set_printoptions(formatter={"float_kind": "{:.3f}".format})

    parser = argparse.ArgumentParser()
    parser.add_argument("--game",
                        required=True,
                        choices=["arbitrary_social_dilemma",
                                 "Functional_form_game",
                                 "Symmetrical_nPD",
                                 "Symmetrical_nCH",
                                 "Symmetrical_nSH",
                                 "Cyclical_nPD",
                                 "Cyclical_nCH",
                                 "Cyclical_nSH",
                                 "Tycoon_nPD",
                                 "Tycoon_nCH",
                                 "Tycoon_nSH",
                                 "Circular_nPD",
                                 "Circular_nCH",
                                 "Circular_nSH",
                                 "too_many_cooks_in_prison",
                                 "PublicGoodsGame"],
                        help="The name of the game to solve")

    args = parser.parse_args()

    payoffs = importlib.import_module("rtm.payoffs")
    game = getattr(payoffs, args.game)

    if args.game == "Functional_form_game":
        n = 5
        print(f"Solving for {args.game} with {n} players")
        nfg = payoffs.generate_matrix(n, game, d=2, c=3)

        s_star = algorithms.find_s_star(nfg)
        T, g_star, _, _ = algorithms.find_T_star(nfg, balance=True)

        print_rtm_info(n, nfg, T, s_star, g_star)

        print("Time to form and solve the game as follows:")
        times = []
        for n in range(8, 18, 1):
            nfg = payoffs.generate_matrix(n, game)

            T, g_star, formulation_time, solver_time = algorithms.find_T_star(nfg, balance=False)
            times.append(formulation_time + solver_time)
            print(
                f"n: {n}, num bytes: {nfg.nbytes}, formulation_time: {formulation_time:.2f}, solver_time: {solver_time:.2f}, total_time: {formulation_time+solver_time:.2f}"
            )

        print(times)

    elif args.game == "arbitrary_social_dilemma":
        print(f"Solving for {args.game}")
        s_star = algorithms.find_s_star(game)
        T, g_star, _, _ = algorithms.find_T_star(game, balance=True)

        print_rtm_info(3, game, T, s_star, g_star)

    elif args.game == "too_many_cooks_in_prison":
        print(f"Solving for {args.game}")
        s_star = 3/5
        T, g_star, _, _ = algorithms.find_T_star(game, action_profile=(0,1,1))

        from fractions import Fraction
        np.set_printoptions(formatter={
            'all': lambda x: str(Fraction(x).limit_denominator(500))
        })

        print_rtm_info(3, game, T, s_star, g_star)

    elif args.game == "PublicGoodsGame":
        print(f"Solving for {args.game}")
        for n in range(2, 6):
            for k in range(1,3):
                nfg = payoffs.generate_matrix(n, game, k=k)

                s_star = algorithms.find_s_star(nfg)
                print(f"n={n}, k={k}, s*={s_star}")

    else:  # one of the network game variants
        for n in range(2, 6):
            print(f"Solving for {args.game} with {n} players")
            nfg = payoffs.generate_matrix(n, game, d=1, c=3)

            s_star = algorithms.find_s_star(nfg)
            T, g_star, _, _ = algorithms.find_T_star(nfg, balance=True)

            print_rtm_info(n, nfg, T, s_star, g_star)
