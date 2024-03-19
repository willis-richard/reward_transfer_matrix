import numpy as np

from rtm import algorithms
from rtm.payoffs import DTYPE

# yapf: disable
nfg = np.array(
    [[(3, 3), (0, 4)],
     [(4, 0), (1, 1)]],
    np.dtype([(f'p{i}', DTYPE) for i in range(2)]))
# yapf: enable

s_star = algorithms.find_s_star(nfg)
np.testing.assert_almost_equal(s_star, 0.75)
T, _, _, _ = algorithms.find_T_star(nfg)
ans = np.array([[0.75, 0.25], [0.25, 0.75]])
np.testing.assert_array_almost_equal(T, ans)

# yapf: disable
nfg = np.array(
    [[(6, 3), (0, 4)],
     [(8, 0), (2, 1)]],
    np.dtype([(f'p{i}', DTYPE) for i in range(2)]))
# yapf: enable

s_star = algorithms.find_s_star(nfg)
np.testing.assert_almost_equal(s_star, 0.6)
T, _, _, _ = algorithms.find_T_star(nfg)
ans = np.array([[0.6, 0.4], [0.4, 0.6]])
np.testing.assert_array_almost_equal(T, ans)
T, _, _, _ = algorithms.find_T_star(nfg, equality=False)
ans = np.array([[0.6, 0.1], [0.4, 0.6]])
np.testing.assert_array_almost_equal(T, ans)

nfg = np.array([[(8, 5), (2, 6)],
                [(10, 2), (4, 3)]],
               np.dtype([(f'p{i}', DTYPE) for i in range(2)]))
# yapf: enable

s_star = algorithms.find_s_star(nfg)
np.testing.assert_almost_equal(s_star, 0.6)
T, _, _, _ = algorithms.find_T_star(nfg)
ans = np.array([[0.6, 0.4], [0.4, 0.6]])
np.testing.assert_array_almost_equal(T, ans)

# Let us try for a 3-player Cyclical-PD
# yapf: disable
nfg = np.array(
    [[[(3, 3, 3), (3, 4, 0)],
      [(4, 0, 3), (4, 1, 0)]],
     [[(0, 3, 4), (0, 4, 1)],
      [(1, 0, 4), (1, 1, 1)]]],
    np.dtype([(f'p{i}', DTYPE) for i in range(3)])
).transpose((1, 2, 0))
# yapf: enable

s_star = algorithms.find_s_star(nfg)
np.testing.assert_almost_equal(s_star, 0.6)
T, _, _, _ = algorithms.find_T_star(nfg)
ans = np.array([[0.75, 0, 0.25], [0.25, 0.75, 0], [0, 0.25, 0.75]])

# 3-player Symmetrical-nPD
# yapf: disable
nfg = np.array(
    [[[(3, 3, 3), (1.5, 4, 1.5)],
      [(4, 1.5, 1.5), (2.5, 2.5, 0)]],
     [[(1.5, 1.5, 4), (0, 2.5, 2.5)],
      [(2.5, 0, 2.5), (1, 1, 1)]]],
             np.dtype([(f'p{i}', DTYPE) for i in range(3)])
).transpose((1, 2, 0))
# yapf: enable

s_star = algorithms.find_s_star(nfg)
np.testing.assert_almost_equal(s_star, 0.6)
T, _, _, _ = algorithms.find_T_star(nfg, balance=True)
ans = np.array([[0.6, 0.2, 0.2], [0.2, 0.6, 0.2], [0.2, 0.2, 0.6]])
np.testing.assert_array_almost_equal(T, ans)
