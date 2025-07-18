import numpy as np

TRIANGULAR = np.array([0,1,3,6,10,15,21,28,36,45])
JIT_ENABLED = False
SAVE_POPULATION = True
EPSILON_FLOAT64 = np.finfo(np.float64).eps
NP_FLOAT_MAX = np.finfo(np.float64).max
NP_FLOAT_MIN = np.finfo(np.float64).min
NUM_THREADS = 12