import numpy as np
from numpy.typing import NDArray

from typing import Dict
from firm_ce.common.constants import JIT_ENABLED
from firm_ce.system.topology import Node_InstanceType

if JIT_ENABLED:
    from numba.core.types import float64, int64
    from numba.experimental import jitclass
    
else:
    def jitclass(spec):
        def decorator(cls):
            return cls
        return decorator

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

    def get_year_t_boundaries(self, year: int) -> NDArray[np.int64]:
        if year < self.year_count - 1:
            last_t = self.year_first_t[year+1]
        else:
            last_t = self.intervals_count
        return self.year_first_t[year], last_t
    
    def set_year_energy_demand(self, nodes_typed_dict: Node_InstanceType) -> None:
        for year in range(self.year_count):
            first_t, last_t = self.get_year_t_boundaries(year)
            for node in nodes_typed_dict.values():
                self.year_energy_demand[year] += sum(node.get_data("trace")[first_t:last_t]) * self.resolution
        return None
    
    def unset_year_energy_demand(self) -> None:
        self.year_energy_demand = np.zeros(self.year_count, dtype=np.float64)
        return None

    def check_reliability_constraint(self, year: int, year_unserved_energy: float) -> bool:
        return (year_unserved_energy / self.year_energy_demand[year]) <= self.allowance

ScenarioParameters_InstanceType = ScenarioParameters.class_type.instance_type

class ModelConfig:
    def __init__(self, config_dict: Dict[str, str]) -> None:
        config_dict = { item['name']: item['value'] for item in config_dict.values() }
        self.type = config_dict['type']
        self.iterations = int(config_dict['iterations'])
        self.population = int(config_dict['population'])
        self.mutation = float(config_dict['mutation'])
        self.recombination = float(config_dict['recombination'])
        self.global_optimal_lcoe = float(config_dict.get('global_optimal_lcoe', 0.0))
        self.near_optimal_tol = float(config_dict.get('near_optimal_tol', 0.0))
        self.midpoint_count = int(config_dict.get('midpoint_count', 0))
        self.balancing_type = str(config_dict['balancing_type'])