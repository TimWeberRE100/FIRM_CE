from firm_ce.common.exceptions import raise_static_modification_error
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
        ('node_count', int64),
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
                 intervals_count,
                 node_count,):        

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

if JIT_ENABLED:
    interval_memory_spec = [
        ('static_instance',boolean),
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
    def __init__(self, static_instance, node_count):
        self.static_instance = static_instance
        self.discharge_max_t = np.zeros(node_count, dtype=np.float64)
        self.charge_max_t = np.zeros(node_count, dtype=np.float64)
        self.flexible_max_t = np.zeros(node_count, dtype=np.float64)
        self.netload_t = np.zeros(node_count, dtype=np.float64)
        
        # Pre-charging
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

if JIT_ENABLED:
    fleetcapacities_spec = [
        ('static_instance',boolean),
        ('generator_power', float64[:]),
        ('generator_newbuild_power', float64[:]),
        ('storage_power', float64[:]),
        ('storage_energy', float64[:]),

        ('flexible_sorted_order', int64[:,:]),
        ('storage_sorted_order', int64[:,:]),

        ('generator_power_nodal', float64[:]),
        ('storage_power_nodal', float64[:]),
        ('storage_energy_nodal', float64[:]),
    ]
else:
    fleetcapacities_spec = []

@jitclass(fleetcapacities_spec)
class FleetCapacities:
    def __init__(self, static_instance):
        self.static_instance = static_instance
        self.generator_power = np.empty(0, dtype=np.float64)
        self.generator_newbuild_power = np.empty(0, dtype=np.float64)
        self.storage_power = np.empty(0, dtype=np.float64)
        self.storage_energy = np.empty(0, dtype=np.float64)

        self.flexible_sorted_order = np.empty((0,0),  dtype=np.int64)
        self.storage_sorted_order = np.empty((0,0),  dtype=np.int64)

        self.generator_power_nodal = np.empty(0, dtype=np.float64)
        self.storage_power_nodal = np.empty(0, dtype=np.float64)
        self.storage_energy_nodal = np.empty(0, dtype=np.float64)

    def allocate_memory(self, 
                        node_count: int,
                        generator_count: int,
                        storage_count: int,
                        flexible_nodal_count_max: int,
                        storage_nodal_count_max: int,) -> None:
        if self.static_instance:
            raise_static_modification_error()
        self.generator_power = np.zeros(generator_count, dtype=np.float64)
        self.generator_newbuild_power = np.zeros(generator_count, dtype=np.float64)
        self.storage_power = np.zeros(storage_count, dtype=np.float64)
        self.storage_energy = np.zeros(storage_count, dtype=np.float64)

        self.flexible_sorted_order = np.full((node_count,flexible_nodal_count_max), -1,  dtype=np.int64)
        self.storage_sorted_order = np.full((node_count,storage_nodal_count_max), -1, dtype=np.int64)
        
        self.generator_power_nodal = np.zeros(node_count, dtype=np.float64)
        self.storage_power_nodal = np.zeros(node_count, dtype=np.float64)
        self.storage_energy_nodal = np.zeros(node_count, dtype=np.float64)

        return None
    
    def build_capacities(self, fleet, new_build_capacities_x):
        if self.static_instance:
            raise_static_modification_error()

        for order, index in enumerate(fleet.generator_x_indices):
            self.generator_power[order] += new_build_capacities_x[index]
            self.generator_newbuild_power[order] += new_build_capacities_x[index]
            self.generator_power_nodal[fleet.generators[order].node.order] += new_build_capacities_x[index]

        for order, index in enumerate(fleet.storage_power_x_indices):
            self.storage_power[order] += new_build_capacities_x[index]
            self.storage_power_nodal[fleet.storages[order].node.order] += new_build_capacities_x[index]

            if fleet.storages[order].duration > 0:
                self.storage_energy[order] += new_build_capacities_x[index] * fleet.storages[order].duration
                self.storage_energy_nodal[fleet.storages[order].node.order] += new_build_capacities_x[index] * fleet.storages[order].duration

        for order, index in enumerate(fleet.storage_energy_x_indices):
            if fleet.storages[order].duration == 0:
                self.storage_energy[order] += new_build_capacities_x[index]
                self.storage_energy_nodal[fleet.storages[order].node.order] += new_build_capacities_x[index]
    
        return None

    def load_data(self, 
                  fleet: Fleet.class_type.instance_type, 
                  node_count: int) -> None:
        self.allocate_memory(node_count,
                             len(fleet.generators),
                             len(fleet.storages),
                             max(fleet.get_generator_nodal_counts(node_count, "flexible")),
                             max(fleet.get_storage_nodal_counts(node_count))
                             )
        for generator in fleet.generators.values():
            self.generator_power[generator.node.order] += generator.capacity

        for storage in fleet.storages.values():
            self.storage_power[storage.node.order] += storage.power_capacity
            self.storage_energy[storage.node.order] += storage.energy_capacity        
        
        #self.get_sorted_orders(fleet)
        
        return None

if JIT_ENABLED:
    energybalance_spec = [
        ('static_instance',boolean),
        ('imports', float64[:,:]), 
        ('exports', float64[:,:]), 
        ('residual_load', float64[:,:]), 
        ('deficits', float64[:,:]), 
        ('spillage', float64[:,:]), 
        ('flexible_power_nodal', float64[:,:]), 
        ('storage_power_nodal', float64[:,:]),
        ('flexible_energy_nodal', float64[:,:]), 
        ('storage_energy_nodal', float64[:,:]),
    ]
else: 
    energybalance_spec = []

@jitclass(energybalance_spec)
class EnergyBalance:
    def __init__(self, static_instance):
        
        self.static_instance = static_instance
        self.imports = np.empty((0, 0), dtype=np.float64)
        self.exports = np.empty((0, 0), dtype=np.float64)
        self.residual_load = np.empty((0, 0), dtype=np.float64)
        self.deficits = np.empty((0, 0), dtype=np.float64)
        self.spillage = np.empty((0, 0), dtype=np.float64)
        self.flexible_power_nodal = np.empty((0, 0), dtype=np.float64)
        self.storage_power_nodal = np.empty((0, 0), dtype=np.float64)
        self.flexible_energy_nodal = np.empty((0, 0), dtype=np.float64)
        self.storage_energy_nodal = np.empty((0, 0), dtype=np.float64)

    def create_dynamic_copy(self):
        return EnergyBalance(False)

    def allocate_memory(self, node_count, intervals_count):
        self.imports = np.zeros((intervals_count, node_count), dtype=np.float64)
        self.exports = np.zeros((intervals_count, node_count), dtype=np.float64)

        self.deficits = np.zeros((intervals_count, node_count), dtype=np.float64)
        self.spillage = np.zeros((intervals_count, node_count), dtype=np.float64)
        self.flexible_power_nodal = np.zeros((intervals_count, node_count), dtype=np.float64)
        self.storage_power_nodal = np.zeros((intervals_count, node_count), dtype=np.float64)
        self.flexible_energy_nodal = np.zeros((intervals_count, node_count), dtype=np.float64)
        self.storage_energy_nodal = np.zeros((intervals_count, node_count), dtype=np.float64)
        return None
    
    def unload_data(self):
        self.imports = np.empty((0, 0), dtype=np.float64)
        self.exports = np.empty((0, 0), dtype=np.float64)
        self.residual_load = np.empty((0, 0), dtype=np.float64)
        self.deficits = np.empty((0, 0), dtype=np.float64)
        self.spillage = np.empty((0, 0), dtype=np.float64)
        self.flexible_power_nodal = np.empty((0, 0), dtype=np.float64)
        self.storage_power_nodal = np.empty((0, 0), dtype=np.float64)
        self.flexible_energy_nodal = np.empty((0, 0), dtype=np.float64)
        self.storage_energy_nodal = np.empty((0, 0), dtype=np.float64)
    
    def initialise_residual_load(self,
                                 generators_typed_dict,
                                 nodes_typed_dict,
                                 intervals_count,):
        self.residual_load = np.zeros((intervals_count,len(nodes_typed_dict)), dtype=np.float64)
        
        # For-loops are faster with Numba than Numpy array operations
        for node in nodes_typed_dict.values():
            for interval in range(intervals_count):
                self.residual_load[interval, node.order] = node.data[interval]

        for generator in generators_typed_dict.values():
            if (generator.data.shape[0] > 0) and (generator.max_build > 0):
                for interval in range(intervals_count):
                    self.residual_load[interval, generator.node.order] -= generator.data[interval] * generator.capacity
        return None

    def update_residual_load(self, 
                            generators_typed_dict,
                            intervals_count,
                            generator_newbuild_power,):
        if self.static_instance:
            raise_static_modification_error()

        for generator in generators_typed_dict.values():
            if (generator.data.shape[0] > 0) and (generator.max_build > 0):
                for interval in range(intervals_count):
                    self.residual_load[interval, generator.node.order] -= generator.data[interval] * generator_newbuild_power[generator.order]
        return None
