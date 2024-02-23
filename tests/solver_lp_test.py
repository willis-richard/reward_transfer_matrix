import numpy as np

from common import DTYPE
from solver_lp import find_e_dash, find_rtm

# yapf: disable
A = np.array(
    [[(3, 3), (0, 4)],
    [(4, 0), (1, 1)]],
    np.dtype([(f'p{i}', DTYPE) for i in range(2)]))
# yapf: enable

e_dash = find_e_dash(A)
np.testing.assert_almost_equal(e_dash, 0.75)
G, _ = find_rtm(A)
ans = np.array([[0.75, 0.25], [0.25, 0.75]])
np.testing.assert_array_almost_equal(G, ans)

# yapf: disable
A = np.array(
    [[(6, 3), (0, 4)],
     [(8, 0), (2, 1)]],
    np.dtype([(f'p{i}', DTYPE) for i in range(2)]))
# yapf: enable

e_dash = find_e_dash(A)
np.testing.assert_almost_equal(e_dash, 0.6)
G, _ = find_rtm(A)
ans = np.array([[0.6, 0.1], [0.4, 0.6]])
np.testing.assert_array_almost_equal(G, ans)

A = np.array([[(8, 5), (2, 6)], [(10, 2), (4, 3)]],
             np.dtype([(f'p{i}', DTYPE) for i in range(2)]))
# yapf: enable

e_dash = find_e_dash(A)
np.testing.assert_almost_equal(e_dash, 0.6)
G, _ = find_rtm(A)
ans = np.array([[0.6, 0.1], [0.4, 0.6]])
np.testing.assert_array_almost_equal(G, ans)

# Let us try for a 3-player Cyclical-PD
# yapf: disable
A = np.array(
    [[[(3, 3, 3), (3, 4, 0)],
      [(4, 0, 3), (4, 1, 0)]],
     [[(0, 3, 4), (0, 4, 1)],
      [(1, 0, 4), (1, 1, 1)]]],
    np.dtype([(f'p{i}', DTYPE) for i in range(3)])
).transpose((1, 2, 0))
# yapf: enable

e_dash = find_e_dash(A)
np.testing.assert_almost_equal(e_dash, 0.6)
G, _ = find_rtm(A)
ans = np.array([[0.75, 0, 0.25], [0.25, 0.75, 0], [0, 0.25, 0.75]])

# 3-player Symmetrical-nPD
# yapf: disable
A = np.array(
    [[[(3, 3, 3), (1.5, 4, 1.5)],
      [(4, 1.5, 1.5), (2.5, 2.5, 0)]],
     [[(1.5, 1.5, 4), (0, 2.5, 2.5)],
      [(2.5, 0, 2.5), (1, 1, 1)]]],
             np.dtype([(f'p{i}', DTYPE) for i in range(3)])).transpose(
                 (1, 2, 0))
# yapf: enable

e_dash = find_e_dash(A)
np.testing.assert_almost_equal(e_dash, 0.6)
G, _ = find_rtm(A, balance=True)
ans = np.array([[0.6, 0.2, 0.2], [0.2, 0.6, 0.2], [0.2, 0.2, 0.6]])
np.testing.assert_array_almost_equal(G, ans)


# game that shows you need a constraint even when a player does not (currently) benefit from defecint
# yapf: disable
A = np.array(
    [[[(3, 3, 3), (3, 4, 0)],
    [(4, 0, 3), (4, 1, 0)]],
    [[(0, 3, 4), (0, 4, 1)],
    [(0, 4, 3), (1, 1, 1)]]],
    np.dtype([(f'p{i}', DTYPE) for i in range(3)])
    ).transpose((1, 2, 0))
# yapf: enable
