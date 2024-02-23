import numpy as np
from scipy.optimize import minimize
from scipy.stats import entropy


def maximise_entropy(G, A_ub_reward, b_ub_reward):
    n = G.shape[0]
    diagonals = np.diag(G)
    mask = np.eye(n, dtype=bool)
    non_diagonal_elements = G[~mask].flatten()
    sum_non_diagonals = np.sum(non_diagonal_elements)

    def recombine_diagonals(x):
        recombined = np.zeros_like(G)
        recombined[~mask] = x
        recombined[mask] = diagonals
        return recombined

    # maximise = - minimise the entropy of non-diagonal elements
    def objective(x):
        # return -np.sum(entropy(recombine_diagonals(x), axis=1))
        # return -np.sum(entropy(x.reshape(n, n-1), axis=1))
        return -entropy(x)

    # Do not increase the sum of the matrix elements, to keep it minimal
    def constraint_total_rows(x):
        return sum_non_diagonals - np.sum(x) + np.finfo(float).eps

    # Original rtm constraints
    def lp_constraints(x):
        return b_ub_reward - A_ub_reward.dot(recombine_diagonals(x).flatten()) + np.finfo(float).eps

    constraints = [{
        'type': 'ineq',
        'fun': constraint_total_rows
    }, {
        'type': 'ineq',
        'fun': lp_constraints
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

    G = recombine_diagonals(res.x)

    return G
