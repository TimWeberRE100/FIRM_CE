import numpy as np
from numpy.typing import NDArray

from typing import Dict
from firm_ce.common.constants import JIT_ENABLED
from firm_ce.common.typing import int64, float64
from firm_ce.common.jit_overload import jitclass

if JIT_ENABLED:
    scenario_parameters_spec = [
        ('resolution', float64),
        ('allowance', float64),
        ('first_year', int64),
        ('final_year', int64),
        ('year_count', int64),
        ('leap_year_count', int64),
        ('year_first_t', int64[:]),
        ('intervals_count', int64),
        ('node_count', int64),
        ('fom_scalar', float64),
        ('year_energy_demand',float64[:]),
    ]
else:
    scenario_parameters_spec = []

@jitclass(scenario_parameters_spec)
class ScenarioParameters:
    def __init__(self,
                 resolution: float, 
                 allowance: float,
                 first_year: int,
                 final_year: int, 
                 year_count: int, 
                 leap_year_count: int, 
                 year_first_t: NDArray[np.int64],
                 intervals_count: int,
                 node_count: int,):        

        self.resolution = resolution # length of time interval in hours
        self.allowance = allowance # % annual demand allowed as unserved energy
        self.first_year = first_year # YYYY
        self.final_year = final_year # YYYY
        self.year_count = year_count 
        self.leap_year_count = leap_year_count
        self.year_first_t = year_first_t
        self.intervals_count = intervals_count
        self.node_count = node_count
        self.fom_scalar = (year_count+leap_year_count/365)/year_count # Scale average annual fom to account for leap days for PLEXOS consistency
        self.year_energy_demand = np.zeros(self.year_count, dtype=np.float64)

if JIT_ENABLED:
    ScenarioParameters_InstanceType = ScenarioParameters.class_type.instance_type
else:
    ScenarioParameters_InstanceType = ScenarioParameters

class ModelConfig:
    def __init__(self, config_dict: Dict[str, str]) -> None:
        config_dict = { item['name']: item['value'] for item in config_dict.values() }
        self.type = config_dict['type']
        self.iterations = int(config_dict['iterations'])
        self.population = int(config_dict['population'])
        self.mutation = float(config_dict['mutation'])
        self.recombination = float(config_dict['recombination'])
        self.near_optimal_tol = float(config_dict.get('near_optimal_tol', 0.0))
        self.midpoint_count = int(config_dict.get('midpoint_count', 0))
        self.balancing_type = str(config_dict['balancing_type'])