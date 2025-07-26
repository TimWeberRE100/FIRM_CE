import numpy as np

JIT_ENABLED = True
SAVE_POPULATION = True
EPSILON_FLOAT64 = np.finfo(np.float64).eps
NP_FLOAT_MAX = np.finfo(np.float64).max
NP_FLOAT_MIN = np.finfo(np.float64).min
PENALTY_MULTIPLIER = 1e6
NUM_THREADS = 6