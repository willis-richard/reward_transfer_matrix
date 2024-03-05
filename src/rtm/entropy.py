import numpy as np
from scipy.optimize import minimize
from scipy.stats import entropy


def maximise_entropy(T, A_ub, b_ub, A_eq, b_eq):
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
        return b_ub- A_ub.dot(recombine_diagonals(x).flatten()) + np.finfo(float).eps

    def lp_eq_constraints(x):
        return b_eq- A_eq.dot(recombine_diagonals(x).flatten())

    constraints = [{
        'type': 'eq',
        'fun': lp_eq_constraints
    }, {
        'type': 'ineq',
        'fun': lp_ub_constraints
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
