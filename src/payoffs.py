import numpy as np
from typing import Tuple

from common import DTYPE


class BaseGame:

    @classmethod
    def payoff_D(cls, idx: Tuple, pid: int, d=1, c=3, k=0):
        pass

    @classmethod
    def payoff_C(cls, idx: Tuple, pid: int, d=1, c=3, k=0):
        pass


def payoff(game: BaseGame, idx: Tuple, pid: int, *args, **kwargs):
    if idx[pid - 1] == 0:
        return game.payoff_C(idx, pid, *args, **kwargs)
    elif idx[pid - 1] == 1:
        return game.payoff_D(idx, pid, *args, **kwargs)
    else:
        assert False, f"Invalid action {idx[pid-1]} taken"


def generate_matrix(n, game):
    M = np.empty([2] * n, dtype=np.dtype([(f'p{i}', DTYPE) for i in range(n)]))
    it = np.nditer(M, flags=['multi_index'])
    while not it.finished:
        idx = tuple(it.multi_index)
        rewards = tuple([payoff(game, idx, i + 1) for i in range(n)])
        M[idx] = rewards
        it.iternext()

    return M


class Tycoon_nPD:

    @classmethod
    def payoff_D(cls, idx: Tuple, pid: int, d=1, c=3, k=0):
        n = len(idx)
        n_D = np.count_nonzero(np.array(idx))
        n_C = n - n_D
        if pid == 1:
            return k + d * (n - 1) + c * n_C
        else:
            return k + d + (c if idx[0] == 0 else 0)

    @classmethod
    def payoff_C(cls, idx: Tuple, pid: int, d=1, c=3, k=0):
        n = len(idx)
        n_D = np.count_nonzero(np.array(idx))
        n_C = n - n_D
        if pid == 1:
            return k + c * (n_C - 1)
        else:
            return k + (c if idx[0] == 0 else 0)


class Tycoon_nCH:

    @classmethod
    def payoff_D(cls, idx: Tuple, pid: int, d=1, c=3, k=0):
        n = len(idx)
        n_D = np.count_nonzero(np.array(idx))
        n_C = n - n_D
        if pid == 1:
            return k + (c + d) * n_C
        else:
            return k + ((c + d) if idx[0] == 0 else 0)

    @classmethod
    def payoff_C(cls, idx: Tuple, pid: int, d=1, c=3, k=0):
        n = len(idx)
        n_D = np.count_nonzero(np.array(idx))
        n_C = n - n_D
        if pid == 1:
            return k + c * (n_C - 1) + d * n_D
        else:
            return k + (c if idx[0] == 0 else d)


class Tycoon_nSH:

    @classmethod
    def payoff_D(cls, idx: Tuple, pid: int, d=1, c=3):
        n = len(idx)
        n_D = np.count_nonzero(np.array(idx))
        n_C = n - n_D
        if pid == 1:
            return c * n_C + d * (n_D - 1)
        else:
            return c if idx[0] == 0 else d

    @classmethod
    def payoff_C(cls, idx: Tuple, pid: int, d=1, c=3):
        n = len(idx)
        n_D = np.count_nonzero(np.array(idx))
        n_C = n - n_D
        if pid == 1:
            return (c + d) * (n_C - 1)
        else:
            return c + d if idx[0] == 0 else 0


class Symmetrical_nPD:

    @classmethod
    def payoff_D(cls, idx: Tuple, pid: int, d=1, c=3):
        n = len(idx)
        n_D = np.count_nonzero(np.array(idx))
        n_C = n - n_D
        return d + c * n_C / (n - 1)

    @classmethod
    def payoff_C(cls, idx: Tuple, pid: int, d=1, c=3):
        n = len(idx)
        n_D = np.count_nonzero(np.array(idx))
        n_C = n - n_D
        return c * (n_C - 1) / (n - 1)

class Cyclical_nPD:
    @classmethod
    def payoff_D(cls, idx: Tuple, pid: int, d=1, c=3):
        n = len(idx)
        if idx[pid % n] == 0:
            return d + c
        else:
            return d

    @classmethod
    def payoff_C(cls, idx: Tuple, pid: int, d=1, c=3):
        n = len(idx)
        if idx[pid % n] == 0:
            return c
        else:
            return 0


class Symmetrical_nCH:

    @classmethod
    def payoff_D(cls, idx: Tuple, pid: int, d=1, c=3):
        n = len(idx)
        n_D = np.count_nonzero(np.array(idx))
        n_C = n - n_D
        return (d + c) * n_C / (n - 1)

    @classmethod
    def payoff_C(cls, idx: Tuple, pid: int, d=1, c=3):
        n = len(idx)
        n_D = np.count_nonzero(np.array(idx))
        n_C = n - n_D
        return c * (n_C - 1) / (n - 1) + d * n_D / (n - 1)


class Cyclical_nCH:

    @classmethod
    def payoff_D(cls, idx: Tuple, pid: int, d=1, c=3):
        n = len(idx)
        if idx[pid % n] == 0:
            return d + c
        else:
            return 0

    @classmethod
    def payoff_C(cls, idx: Tuple, pid: int, d=1, c=3):
        n = len(idx)
        if idx[pid % n] == 0:
            return c
        else:
            return d


class Symmetrical_nSH:

    @classmethod
    def payoff_D(cls, idx: Tuple, pid: int, d=1, c=3):
        n = len(idx)
        n_D = np.count_nonzero(np.array(idx))
        n_C = n - n_D
        return c * n_C / (n - 1) + d * (n_D - 1) / (n - 1)

    @classmethod
    def payoff_C(cls, idx: Tuple, pid: int, d=1, c=3):
        n = len(idx)
        n_D = np.count_nonzero(np.array(idx))
        n_C = n - n_D
        return (c + d) * (n_C - 1) / (n - 1)


class Cyclical_nSH:

    @classmethod
    def payoff_D(cls, idx: Tuple, pid: int, d=1, c=3, k=0):
        n = len(idx)
        if idx[pid % n] == 0:
            return k + c
        else:
            return k + 1

    @classmethod
    def payoff_C(cls, idx: Tuple, pid: int, d=1, c=3, k=0):
        n = len(idx)
        if idx[pid % n] == 0:
            return k + c + d
        else:
            return k + 0


class Teams_nPD:

    @classmethod
    def payoff_D(cls, idx: Tuple, pid: int, d=1, c_t=2, c_o=1):
        n_o = math.ceil(len(idx) / 2)
        n_e = math.floor(len(idx) / 2)
        n_D_o = np.count_nonzero(np.array(idx[0::2]))
        n_D_e = np.count_nonzero(np.array(idx[1::2]))
        n_C_o = n_o - n_D_o
        n_C_e = n_e - n_D_e
        player_action = idx[pid - 1]
        partner_action = idx[(pid - 2) if pid % 2 == 0 else pid]
        defect_reward = d
        if pid % 2 == 0:
            cooperation_reward = c_o * n_C_o / n_e + c_t * n_C_e / (n_e - 1)
        else:
            cooperation_reward = c_o * n_C_e / n_o + c_t * n_C_o / (n_o - 1)
        return defect_reward + cooperation_reward

    @classmethod
    def payoff_C(cls, idx: Tuple, pid: int, d=1, c_t=2, c_o=1):
        n_o = math.ceil(len(idx) / 2)
        n_e = math.floor(len(idx) / 2)
        n_D_o = np.count_nonzero(np.array(idx[0::2]))
        n_D_e = np.count_nonzero(np.array(idx[1::2]))
        n_C_o = n_o - n_D_o
        n_C_e = n_e - n_D_e
        player_action = idx[pid - 1]
        partner_action = idx[(pid - 2) if pid % 2 == 0 else pid]
        defect_reward = 0
        if pid % 2 == 0:  # even team
            cooperation_reward = c_o * n_C_o / n_e + c_t * (n_C_e - 1) / (n_e -
                                                                          1)
        else:
            cooperation_reward = c_o * n_C_e / n_o + c_t * (n_C_o - 1) / (n_o -
                                                                          1)
        return defect_reward + cooperation_reward


class Teams_nCH:

    @classmethod
    def payoff_D(cls, idx: Tuple, pid: int, d=1, c_t=2, c_o=1):
        n_o = math.ceil(len(idx) / 2)
        n_e = math.floor(len(idx) / 2)
        n_D_o = np.count_nonzero(np.array(idx[0::2]))
        n_D_e = np.count_nonzero(np.array(idx[1::2]))
        n_C_o = n_o - n_D_o
        n_C_e = n_e - n_D_e
        player_action = idx[pid - 1]
        partner_action = idx[(pid - 2) if pid % 2 == 0 else pid]
        defect_reward = d if player_action != partner_action else 0
        if pid % 2 == 0:
            cooperation_reward = c_o * n_C_o / n_e + c_t * n_C_e / (n_e - 1)
        else:
            cooperation_reward = c_o * n_C_e / n_o + c_t * n_C_o / (n_o - 1)
        return defect_reward + cooperation_reward

    @classmethod
    def payoff_C(cls, idx: Tuple, pid: int, d=1, c_t=2, c_o=1):
        n_o = math.ceil(len(idx) / 2)
        n_e = math.floor(len(idx) / 2)
        n_D_o = np.count_nonzero(np.array(idx[0::2]))
        n_D_e = np.count_nonzero(np.array(idx[1::2]))
        n_C_o = n_o - n_D_o
        n_C_e = n_e - n_D_e
        player_action = idx[pid - 1]
        partner_action = idx[(pid - 2) if pid % 2 == 0 else pid]
        defect_reward = d if player_action != partner_action else 0
        if pid % 2 == 0:  # even team
            cooperation_reward = c_o * n_C_o / n_e + c_t * (n_C_e - 1) / (n_e -
                                                                          1)
        else:
            cooperation_reward = c_o * n_C_e / n_o + c_t * (n_C_o - 1) / (n_o -
                                                                          1)
        return defect_reward + cooperation_reward


class Teams_nSH:

    @classmethod
    def payoff_D(cls, idx: Tuple, pid: int, d=1, c_t=2, c_o=1):
        n_o = math.ceil(len(idx) / 2)
        n_e = math.floor(len(idx) / 2)
        n_D_o = np.count_nonzero(np.array(idx[0::2]))
        n_D_e = np.count_nonzero(np.array(idx[1::2]))
        n_C_o = n_o - n_D_o
        n_C_e = n_e - n_D_e
        player_action = idx[pid - 1]
        partner_action = idx[(pid - 2) if pid % 2 == 0 else pid]
        defect_reward = d if player_action == partner_action else 0
        if pid % 2 == 0:
            cooperation_reward = c_o * n_C_o / n_e + c_t * n_C_e / (n_e - 1)
        else:
            cooperation_reward = c_o * n_C_e / n_o + c_t * n_C_o / (n_o - 1)
        return defect_reward + cooperation_reward

    @classmethod
    def payoff_C(cls, idx: Tuple, pid: int, d=1, c_t=2, c_o=1):
        n_o = math.ceil(len(idx) / 2)
        n_e = math.floor(len(idx) / 2)
        n_D_o = np.count_nonzero(np.array(idx[0::2]))
        n_D_e = np.count_nonzero(np.array(idx[1::2]))
        n_C_o = n_o - n_D_o
        n_C_e = n_e - n_D_e
        player_action = idx[pid - 1]
        partner_action = idx[(pid - 2) if pid % 2 == 0 else pid]
        defect_reward = d if player_action == partner_action else 0
        if pid % 2 == 0:  # even team
            cooperation_reward = c_o * n_C_o / n_e + c_t * (n_C_e - 1) / (n_e -
                                                                          1)
        else:
            cooperation_reward = c_o * n_C_e / n_o + c_t * (n_C_o - 1) / (n_o -
                                                                          1)
        return defect_reward + cooperation_reward


class Local_nPD:

    @classmethod
    def payoff_D(cls, idx: Tuple, pid: int, d=1, c=3):
        n = len(idx)
        n_D = np.count_nonzero(np.array(idx))
        n_C = n - n_D
        pids = np.arange(1, n + 1, 1)
        distance = np.array(
            [min(abs(pid - p), n - abs(pid - p)) for p in pids])

        distance[pid - 1] = 1
        # weight = 1 / distance
        weight = (1.0 / 2)**distance
        weight[pid - 1] = 0

        # weight = weight / np.sum(weight)
        r_v_c = d + c
        r_v_d = d
        values = np.array(idx)
        c_idx = values == 0
        d_idx = values == 1
        values[c_idx] = r_v_c
        values[d_idx] = r_v_d
        # return d + np.dot(values, weight)
        return np.dot(values, weight)

    @classmethod
    def payoff_C(cls, idx: Tuple, pid: int, d=1, c=3):
        n = len(idx)
        n_D = np.count_nonzero(np.array(idx))
        n_C = n - n_D
        pids = np.arange(1, n + 1, 1)
        distance = np.array(
            [min(abs(pid - p), n - abs(pid - p)) for p in pids])

        distance[pid - 1] = 1
        # weight = 1 / distance
        weight = (1.0 / 2)**distance
        weight[pid - 1] = 0

        # weight = weight / np.sum(weight)
        r_v_c = c
        r_v_d = 0
        values = np.array(idx)
        c_idx = values == 0
        d_idx = values == 1
        values[c_idx] = r_v_c
        values[d_idx] = r_v_d
        return np.dot(values, weight)


class Local_nCH:

    @classmethod
    def payoff_D(cls, idx: Tuple, pid: int, d=1, c=3):
        n = len(idx)
        n_D = np.count_nonzero(np.array(idx))
        n_C = n - n_D
        pids = np.arange(1, n + 1, 1)
        distance = np.array(
            [min(abs(pid - p), n - abs(pid - p)) for p in pids])

        distance[pid - 1] = 1
        # weight = 1 / distance
        weight = (1.0 / 2)**distance
        weight[pid - 1] = 0

        # weight = weight / np.sum(weight)
        r_v_c = d + c
        r_v_d = 0
        values = np.array(idx)
        c_idx = values == 0
        d_idx = values == 1
        values[c_idx] = r_v_c
        values[d_idx] = r_v_d
        # return d + np.dot(values, weight)
        return np.dot(values, weight)

    @classmethod
    def payoff_C(cls, idx: Tuple, pid: int, d=1, c=3):
        n = len(idx)
        n_D = np.count_nonzero(np.array(idx))
        n_C = n - n_D
        pids = np.arange(1, n + 1, 1)
        distance = np.array(
            [min(abs(pid - p), n - abs(pid - p)) for p in pids])

        distance[pid - 1] = 1
        # weight = 1 / distance
        weight = (1.0 / 2)**distance
        weight[pid - 1] = 0

        # weight = weight / np.sum(weight)
        r_v_c = c
        r_v_d = d
        values = np.array(idx)
        c_idx = values == 0
        d_idx = values == 1
        values[c_idx] = r_v_c
        values[d_idx] = r_v_d
        return np.dot(values, weight)


class Local_nSH:

    @classmethod
    def payoff_D(cls, idx: Tuple, pid: int, d=1, c=3):
        n = len(idx)
        n_D = np.count_nonzero(np.array(idx))
        n_C = n - n_D
        pids = np.arange(1, n + 1, 1)
        distance = np.array(
            [min(abs(pid - p), n - abs(pid - p)) for p in pids])

        distance[pid - 1] = 1
        # weight = 1 / distance
        weight = (1.0 / 2)**distance
        weight[pid - 1] = 0

        # weight = weight / np.sum(weight)
        r_v_c = c
        r_v_d = d
        values = np.array(idx)
        c_idx = values == 0
        d_idx = values == 1
        values[c_idx] = r_v_c
        values[d_idx] = r_v_d
        # return d + np.dot(values, weight)
        return np.dot(values, weight)

    @classmethod
    def payoff_C(cls, idx: Tuple, pid: int, d=1, c=3):
        n = len(idx)
        n_D = np.count_nonzero(np.array(idx))
        n_C = n - n_D
        pids = np.arange(1, n + 1, 1)
        distance = np.array(
            [min(abs(pid - p), n - abs(pid - p)) for p in pids])

        distance[pid - 1] = 1
        # weight = 1 / distance
        weight = (1.0 / 2)**distance
        weight[pid - 1] = 0

        # weight = weight / np.sum(weight)
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
A = np.array(
    [[[(9, 6, 7), (2, 9, 7)],
    [(8, 4, 8), (3, 2, 1)]],
    [[(1, 6, 12), (0, 5, 2)],
    [(8, 2, 8), (1, 2, 0)]]],
    np.dtype([(f'p{i}', DTYPE) for i in range(3)])
    ).transpose((1, 2, 0))
# yapf: enable


class Functional_form_game:
    # d is the relative weight of the social welfare to a defecting agent vs cooperating
    @classmethod
    def payoff_D(cls, idx: Tuple, pid: int, d=2, c=3):
        n = len(idx)
        n_D = np.count_nonzero(np.array(idx))
        n_C = n - n_D

        sw = -c / n * n_C**2 + 2 * c * n_C
        total_weight = np.sum((d * np.array(idx) + 1) * np.arange(1, n + 1))

        return d * pid * sw / total_weight

    @classmethod
    def payoff_C(cls, idx: Tuple, pid: int, d=2, c=3):
        n = len(idx)
        n_D = np.count_nonzero(np.array(idx))
        n_C = n - n_D

        sw = -c / n * n_C**2 + 2 * c * n_C
        total_weight = np.sum((d * np.array(idx) + 1) * np.arange(1, n + 1))

        return pid * sw / total_weight


class Winner_nPD:
    # Defecting gains 1
    # the player at index n_C gets c*n_C
    # SW = n_D + 2*n_C
    @classmethod
    def payoff_D(cls, idx: Tuple, pid: int, d=1, c=2):
        n = len(idx)
        n_D = np.count_nonzero(np.array(idx))
        n_C = n - n_D
        return d + (c * n_C if n_C == pid else 0)

    @classmethod
    def payoff_C(cls, idx: Tuple, pid: int, d=1, c=2):
        n = len(idx)
        n_D = np.count_nonzero(np.array(idx))
        n_C = n - n_D
        return c * n_C if n_C == pid else 0


class Ladder_connections:

    @classmethod
    def payoff_D(cls, idx: Tuple, pid: int):
        n = len(idx)
        assert n % 2 == 0
        h = int(n / 2)
        connections = np.concatenate((h - np.arange(h), 1 + np.arange(h)))
        # work out what proportion of their rewards they share
        C_rewards = (np.array(idx) == 0).astype(int) * (3.0 / connections)
        # if in 2nd half, get rewards from first half
        if pid > h:
            return 1 + np.sum(C_rewards[0:pid - h])
        else:
            return 1 + np.sum(C_rewards[h + pid - 1:n])

    @classmethod
    def payoff_C(cls, idx: Tuple, pid: int):
        n = len(idx)
        assert n % 2 == 0
        h = int(n / 2)
        connections = np.concatenate((h - np.arange(h), 1 + np.arange(h)))
        # work out what proportion of their rewards they share
        C_rewards = (np.array(idx) == 0).astype(int) * (3.0 / connections)
        # if in 2nd half, get rewards from first half
        if pid > h:
            return np.sum(C_rewards[0:pid - h])
        else:
            return np.sum(C_rewards[h + pid - 1:n])


class Odds_only_evens_only_nPD:
    # use only 4p
    @classmethod
    def payoff_D(cls, idx: Tuple, pid: int, d=1, c_o=3, c_e=2):
        assert len(idx) == 4
        if pid % 2 == 0:  # even team
            n_D = np.count_nonzero(np.array(idx[1::2]))
            n_C = 2 - n_D
            return d + c_e * n_C
        else:
            n_D = np.count_nonzero(np.array(idx[0::2]))
            n_C = 2 - n_D
            return d + c_o * n_C

    @classmethod
    def payoff_C(cls, idx: Tuple, pid: int, d=1, c_o=3, c_e=2):
        assert len(idx) == 4
        if pid % 2 == 0:  # even team
            n_D = np.count_nonzero(np.array(idx[1::2]))
            n_C = 2 - n_D
            return c_e * (n_C - 1)
        else:
            n_D = np.count_nonzero(np.array(idx[0::2]))
            n_C = 2 - n_D
            return c_o * (n_C - 1)


class Symmetrical_nPD_with_bonus_to_ith_player_defecting:

    @classmethod
    def payoff_D(cls, idx: Tuple, pid: int, d=1, c=3):
        n = len(idx)
        n_D = np.count_nonzero(np.array(idx))
        n_C = n - n_D
        bonus = 1 if n_D == pid else 0
        return d + c * n_C / (n - 1) + bonus

    @classmethod
    def payoff_C(cls, idx: Tuple, pid: int, d=1, c=3):
        n = len(idx)
        n_D = np.count_nonzero(np.array(idx))
        n_C = n - n_D
        return c * (n_C - 1) / (n - 1)


class Scaled_nPD:
    # returns scaled by pid
    @classmethod
    def payoff_D(cls, idx: Tuple, pid: int, d=1, c=3):
        n = len(idx)
        n_D = np.count_nonzero(np.array(idx))
        n_C = n - n_D
        return pid * (d + c * n_C / (n - 1))

    @classmethod
    def payoff_C(cls, idx: Tuple, pid: int, d=1, c=3):
        n = len(idx)
        n_D = np.count_nonzero(np.array(idx))
        n_C = n - n_D
        return pid * c * (n_C - 1) / (n - 1)

if __name__ == "__main__":
    from itertools import product
    import numpy as np
    np.set_printoptions(formatter={'float_kind': "{:.2f}".format})
    n = 6
    game = Ladder_connections
    for c in product([0, 1], repeat=n):
        print(c, np.array([payoff(game, c, i + 1) for i in range(n)]))
