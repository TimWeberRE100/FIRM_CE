from firm_ce.common.constants import JIT_ENABLED
from firm_ce.system.components import Fleet
import numpy as np


if JIT_ENABLED:
    from numba.core.types import float64, int64, string, boolean, DictType, UniTuple
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
        ('fom_scalar', float64),
    ]
else:
    scenario_parameters_spec = []

@jitclass(scenario_parameters_spec)
class ScenarioParameters:
    def __init__(self,
                 resolution, 
                 allowance,
                 first_year,
                 final_year, 
                 year_count, 
                 leap_year_count, 
                 year_first_t,
                 intervals_count,):        

        self.resolution = resolution # length of time interval in hours
        self.allowance = allowance # % annual demand allowed as unserved energy
        self.first_year = first_year # YYYY
        self.final_year = final_year # YYYY
        self.year_count = year_count 
        self.leap_year_count = leap_year_count
        self.year_first_t = year_first_t
        self.intervals_count = intervals_count
        self.fom_scalar = (year_count+leap_year_count/365)/year_count # Scale average annual fom to account for leap days for PLEXOS consistency

if JIT_ENABLED:
    interval_memory_spec = [
        ('discharge_max_t', float64[:]),
        ('charge_max_t', float64[:]),
        ('flexible_max_t', float64[:]),
        ('netload_t', float64[:]),
        ('deficit_block_min', float64[:]),
        ('deficit_block_max', float64[:]),
        ('storage_reversed_t', float64[:]),
        ('flexible_limit_reversed_t', float64[:]),
        ('precharge_energy', float64[:]),
        ('storage_trickling_reserves', float64[:]),
        ('flexible_trickling_reserves', float64[:]),
        ('precharge_mask', boolean[:]),
        ('storage_trickling_mask', boolean[:]),
        ('flexible_trickling_mask', boolean[:]),
    ]
else:
    interval_memory_spec = []

@jitclass(interval_memory_spec)
class IntervalMemory:
    def __init__(self):
        self.discharge_max_t = np.empty(0, dtype=np.float64)
        self.charge_max_t = np.empty(0, dtype=np.float64)
        self.flexible_max_t = np.empty(0, dtype=np.float64)
        self.netload_t = np.empty(0, dtype=np.float64)

        # Pre-charging
        self.deficit_block_min = np.empty(0, dtype=np.float64)
        self.deficit_block_max = np.empty(0, dtype=np.float64)
        self.storage_reversed_t = np.empty(0, dtype=np.float64)
        self.flexible_limit_reversed_t = np.empty(0, dtype=np.float64)

        self.precharge_energy = np.empty(0, dtype=np.float64)
        self.storage_trickling_reserves = np.empty(0, dtype=np.float64)
        self.flexible_trickling_reserves = np.empty(0, dtype=np.float64)
        self.precharge_mask = np.empty(0, dtype=np.bool_)
        self.storage_trickling_mask = np.empty(0, dtype=np.bool_)
        self.flexible_trickling_mask = np.empty(0, dtype=np.bool_)

    def allocate_memory(self, node_count: int) -> None:
        self.discharge_max_t = np.zeros(node_count, dtype=np.float64)
        self.charge_max_t = np.zeros(node_count, dtype=np.float64)
        self.flexible_max_t = np.zeros(node_count, dtype=np.float64)
        self.netload_t = np.zeros(node_count, dtype=np.float64)
        self.deficit_block_min = np.zeros(node_count, dtype=np.float64)
        self.deficit_block_max = np.zeros(node_count, dtype=np.float64)
        self.storage_reversed_t = np.zeros(node_count, dtype=np.float64)
        self.flexible_limit_reversed_t = np.zeros(node_count, dtype=np.float64)
        self.precharge_energy = np.zeros(node_count, dtype=np.float64)
        self.storage_trickling_reserves = np.zeros(node_count, dtype=np.float64)
        self.flexible_trickling_reserves = np.zeros(node_count, dtype=np.float64)

        self.precharge_mask = np.zeros(node_count, dtype=np.bool_)
        self.storage_trickling_mask = np.zeros(node_count, dtype=np.bool_)
        self.flexible_trickling_mask = np.zeros(node_count, dtype=np.bool_)

        return None

if JIT_ENABLED:
    fleetcapacities_spec = [
        ('solar', float64[:]),
        ('wind', float64[:]),
        ('baseload', float64[:]),
        ('flexible', float64[:]),
        ('storage_power', float64[:]),
        ('storage_energy', float64[:]),
    ]
else:
    fleetcapacities_spec = []

@jitclass(fleetcapacities_spec)
class FleetCapacities:
    def __init__(self):
        ###### DO I ACTUALLY NEED SOLAR, WIND AND BASELOAD HERE?
        self.solar = np.empty(0, dtype=np.float64)
        self.wind = np.empty(0, dtype=np.float64)
        self.baseload = np.empty(0, dtype=np.float64)
        self.flexible = np.empty(0, dtype=np.float64)
        self.storage_power = np.empty(0, dtype=np.float64)
        self.storage_energy = np.empty(0, dtype=np.float64)

    def allocate_memory(self, node_count: int) -> None:
        self.solar = np.zeros(node_count, dtype=np.float64)
        self.wind = np.zeros(node_count, dtype=np.float64)
        self.baseload = np.zeros(node_count, dtype=np.float64)
        self.flexible = np.zeros(node_count, dtype=np.float64)
        self.storage_power = np.zeros(node_count, dtype=np.float64)
        self.storage_energy = np.zeros(node_count, dtype=np.float64)
        return None

    def load_data(self, fleet: Fleet.class_type.instance_type, node_count: int) -> None:
        self.allocate_memory(node_count)
        for generator in fleet.generators.values():
            if generator.unit_type == 'solar':
                self.solar[generator.node.order] += generator.capacity
            elif generator.unit_type == 'wind':
                self.wind[generator.node.order] += generator.capacity
            elif generator.unit_type == 'baseload':
                self.baseload[generator.node.order] += generator.capacity
            elif generator.unit_type == 'flexible':
                self.flexible[generator.node.order] += generator.capacity

        for storage in fleet.storages.values():
            self.storage_power[storage.node.order] += storage.power_capacity
            self.storage_energy[storage.node.order] += storage.energy_capacity
        return None

if JIT_ENABLED:
    energybalance_spec = [
        ('interval_memory', IntervalMemory.class_type.instance_type),
        ('fleet_capacities', FleetCapacities.class_type.instance_type),
        ('imports', float64[:,:]), 
        ('exports', float64[:,:]), 
        ('residual_load', float64[:,:]), 
        ('deficits', float64[:,:]), 
        ('spillage', float64[:,:]), 
        ('flexible_power_nodal', float64[:,:]), 
        ('storage_power_nodal', float64[:,:]),
        ('flexible_energy_nodal', float64[:,:]), 
        ('storage_energy_nodal', float64[:,:]),
        ('flexible_sorted_order', int64[:,:]),
        ('storage_sorted_order', int64[:,:]),
    ]
else: 
    energybalance_spec = []

@jitclass(energybalance_spec)
class EnergyBalance:
    def __init__(self,
                 interval_memory,
                 fleet_capacities,
                 imports, 
                 exports, 
                 residual_load, 
                 deficits, 
                 spillage, 
                 flexible_power_nodal, 
                 storage_power_nodal,
                 flexible_energy_nodal,
                 storage_energy_nodal,
                 flexible_sorted_order,
                 storage_sorted_order,):
        
        self.interval_memory = interval_memory
        self.fleet_capacities = fleet_capacities
        self.imports = imports
        self.exports = exports
        self.residual_load = residual_load
        self.deficits = deficits
        self.spillage = spillage
        self.flexible_power_nodal = flexible_power_nodal
        self.storage_power_nodal = storage_power_nodal
        self.flexible_energy_nodal = flexible_energy_nodal
        self.storage_energy_nodal = storage_energy_nodal
        self.flexible_sorted_order = flexible_sorted_order
        self.storage_sorted_order = storage_sorted_order

    def allocate_memory(self, node_count):
        self.interval_memory.allocate_memory(node_count)        

    def load_data(self, fleet, node_count):  
        self.allocate_memory(node_count)      
        self.fleet_capacities.load_data(fleet, node_count)
        return None
    
    def calculate_residual_load(self, 
                                generators_typed_dict,
                                nodes_typed_dict,
                                intervals_count,):
        self.residual_load = np.zeros((intervals_count,len(nodes_typed_dict)), dtype=np.float64)
        
        for node in nodes_typed_dict.values():
            self.residual_load[node.order] = node.data

        for generator in generators_typed_dict.values():
            if generator.data.shape[0] > 0:
                self.residual_load[generator.node.order] -= generator.data

        return None
