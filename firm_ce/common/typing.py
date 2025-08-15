import numpy as np
from numpy.typing import NDArray
from typing import Tuple, Dict, List

from firm_ce.system.parameters import ScenarioParameters_InstanceType
from firm_ce.system.topology import Network_InstanceType
from firm_ce.system.components import Fleet_InstanceType

EvaluationRecord_Type = Tuple[str, str, float, float, float, NDArray[np.float64]]
DifferentialEvolutionArgs_Type = Tuple[ScenarioParameters_InstanceType, Fleet_InstanceType, Network_InstanceType, str]
BroadOptimumVars_Type = Tuple[int, str, bool, str]
BandCandidates_Type = Dict[str, Tuple[List[float], List[float]]]