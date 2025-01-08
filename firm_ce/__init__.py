import numpy as np

from firm_ce import (
    file_manager,
    components,
    optimisation
)

from firm_ce.model import (
    Model,
    Scenario,
    ModelConfig
)

TRIANGULAR = np.array([0,1,3,6,10,15,21,28,36])