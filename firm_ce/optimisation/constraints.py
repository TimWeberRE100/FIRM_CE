import numpy as np

from firm_ce.constants import JIT_ENABLED

if JIT_ENABLED:
    from numba import njit
else:
    def njit(func=None, **kwargs):
        if func is not None:
            return func
        def wrapper(f):
            return f
        return wrapper

@njit
def nodal_monotonic_constraint(x, group_sizes, offset):
    total_diffs = 0
    for i in range(group_sizes.shape[0]):
        total_diffs += group_sizes[i] - 1

    # If differential evolution is not vectorised
    if x.ndim == 1:
        diffs = np.empty(total_diffs, dtype=x.dtype)
        idx = 0
        pos = 0
        for i in range(group_sizes.shape[0]):
            size = group_sizes[i]
            for j in range(size - 1):
                diffs[idx] = x[offset + pos + j + 1] - x[offset + pos + j]
                idx += 1
            pos += size
        return diffs
    
    # If differential evolution is vectorised
    else:
        S = x.shape[1]
        diffs = np.empty((total_diffs, S), dtype=x.dtype)
        for s in range(S):
            idx = 0
            pos = 0
            for i in range(group_sizes.shape[0]):
                size = group_sizes[i]
                for j in range(size - 1):
                    diffs[idx, s] = x[offset + pos + j + 1, s] - x[offset + pos + j, s]
                    idx += 1
                pos += size
        return diffs

class BalancingMonotonicityConstraint:
    def __init__(self, group_sizes, offset):
        self.group_sizes = np.array(group_sizes, dtype=np.int64)
        self.offset = offset
        self.total_diffs = int(np.sum(self.group_sizes - 1))
        self.lb = np.zeros(self.total_diffs)
        self.ub = np.full(self.total_diffs, np.inf)

    def __call__(self, x):
        return nodal_monotonic_constraint(x, self.group_sizes, self.offset)