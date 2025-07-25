import numpy as np
from numpy.typing import NDArray

from firm_ce.common.exceptions import raise_static_modification_error
from firm_ce.common.constants import JIT_ENABLED
from firm_ce.system.components import Fleet_InstanceType, Generator_InstanceType, Storage_InstanceType
from firm_ce.system.topology import Node_InstanceType
import firm_ce.common.helpers as helpers

if JIT_ENABLED:
    from numba.core.types import float64, int64, boolean
    from numba.typed.typeddict import Dict as TypedDict
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
        self.year_energy_demand = np.zeros(self.year_count, dtype=np.float64)

    def get_year_t_boundaries(self, year):
        if year < self.year_count - 1:
            last_t = self.year_first_t[year+1]-1
        else:
            last_t = self.intervals_count
        return self.year_first_t[year], last_t
    
    def set_year_energy_demand(self, nodes_typed_dict: Node_InstanceType) -> None:
        for year in range(self.year_count):
            first_t, last_t = self.get_year_t_boundaries(year)
            for node in nodes_typed_dict.values():
                self.year_energy_demand[year] += sum(node.get_data()[first_t:last_t]) * self.resolution
        return None
    
    def unset_year_energy_demand(self) -> None:
        self.year_energy_demand = np.zeros(self.year_count, dtype=np.float64)
        return None

    def check_reliability_constraint(self, year: int, year_unserved_energy: float) -> bool:
        return (year_unserved_energy / self.year_energy_demand[year]) <= self.allowance

if JIT_ENABLED:
    fleetcapacities_spec = [
        ('static_instance',boolean),
        ('generator_power', float64[:]),
        ('generator_newbuild_power', float64[:]),
        
        ('flexible_power', float64[:]),
        ('storage_power', float64[:]),
        ('storage_energy', float64[:]),
        ('storage_d_efficiencies', float64[:]),
        ('storage_c_efficiencies', float64[:]),

        ('flexible_sorted_order', int64[:,:]),
        ('storage_sorted_order', int64[:,:]),
        ('flexible_node_orders', int64[:]),
        ('storage_node_orders', int64[:]),

        ('flexible_power_nodal', float64[:]),
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

        self.flexible_power = np.empty(0, dtype=np.float64)
        self.storage_power = np.empty(0, dtype=np.float64)
        self.storage_energy = np.empty(0, dtype=np.float64)
        self.storage_d_efficiencies = np.empty(0, dtype=np.float64)
        self.storage_c_efficiencies = np.empty(0, dtype=np.float64)

        self.flexible_sorted_order = np.empty((0,0),  dtype=np.int64)
        self.storage_sorted_order = np.empty((0,0),  dtype=np.int64)
        self.flexible_node_orders = np.empty(0,  dtype=np.int64)
        self.storage_node_orders = np.empty(0,  dtype=np.int64)

        self.generator_power_nodal = np.empty(0, dtype=np.float64)
        self.flexible_power_nodal = np.empty(0, dtype=np.float64)
        self.storage_power_nodal = np.empty(0, dtype=np.float64)
        self.storage_energy_nodal = np.empty(0, dtype=np.float64)

    def allocate_memory(self, 
                        node_count: int,
                        generator_count: int,
                        flexible_count: int,
                        storage_count: int,
                        flexible_nodal_count_max: int,
                        storage_nodal_count_max: int,) -> None:
        if self.static_instance:
            raise_static_modification_error()
        self.generator_power = np.zeros(generator_count, dtype=np.float64)
        self.generator_newbuild_power = np.zeros(generator_count, dtype=np.float64)

        self.flexible_power = np.zeros(flexible_count, dtype=np.float64)
        self.storage_power = np.zeros(storage_count, dtype=np.float64)
        self.storage_energy = np.zeros(storage_count, dtype=np.float64)
        self.storage_d_efficiencies = np.zeros(storage_count, dtype=np.float64)
        self.storage_c_efficiencies = np.zeros(storage_count, dtype=np.float64)

        self.flexible_sorted_order = np.full((node_count,flexible_nodal_count_max), -1,  dtype=np.int64)
        self.storage_sorted_order = np.full((node_count,storage_nodal_count_max), -1, dtype=np.int64)
        self.flexible_node_orders = np.full(flexible_count, -1, dtype=np.int64)
        self.storage_node_orders = np.full(storage_count, -1, dtype=np.int64)

        self.generator_power_nodal = np.zeros(node_count, dtype=np.float64)
        self.flexible_power_nodal = np.zeros(node_count, dtype=np.float64)
        self.storage_power_nodal = np.zeros(node_count, dtype=np.float64)
        self.storage_energy_nodal = np.zeros(node_count, dtype=np.float64)

        return None

    def load_data(self, 
                  fleet: Fleet_InstanceType, 
                  node_count: int) -> None:
        self.allocate_memory(node_count,
                             len(fleet.generators),
                             fleet.get_generator_unit_type_count('flexible'),
                             len(fleet.storages),
                             max(fleet.get_generator_nodal_counts(node_count, "flexible")),
                             max(fleet.get_storage_nodal_counts(node_count))
                             )
        for order, generator in fleet.generators.items():
            self.generator_power[order] += generator.capacity
            self.generator_power_nodal[generator.node.order] += generator.capacity

            if generator.unit_type == "flexible":
                self.flexible_power[order] += generator.capacity
                self.flexible_power_nodal[generator.node.order] += generator.capacity

        for order, storage in fleet.storages.items():
            self.storage_power[order] += storage.power_capacity
            self.storage_energy[order] += storage.energy_capacity 
            self.storage_power_nodal[storage.node.order] += storage.power_capacity
            self.storage_energy_nodal[storage.node.order] += storage.energy_capacity

        self.storage_d_efficiencies, self.storage_c_efficiencies = fleet.get_storage_efficiency_arrays()       
        self.storage_node_orders = fleet.get_storage_node_order_array()
        self.flexible_node_orders = fleet.get_generator_node_order_array('flexible')
        #self.get_sorted_orders(fleet)
        
        return None
    
    def build_capacities(self, fleet, new_build_capacities_x):
        if self.static_instance:
            raise_static_modification_error()

        for order, index in enumerate(fleet.generator_x_indices):
            self.generator_power[order] += new_build_capacities_x[index]
            self.generator_newbuild_power[order] += new_build_capacities_x[index]
            self.generator_power_nodal[fleet.generators[order].node.order] += new_build_capacities_x[index]

            if fleet.generators[order].unit_type == "flexible":
                self.flexible_power[order] += new_build_capacities_x[index]
                self.flexible_power_nodal[fleet.generators[order].node.order] += new_build_capacities_x[index]

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

if JIT_ENABLED:
    interval_memory_spec = [
        ('static_instance',boolean),
        ('interval',int64),
        ('discharge_max_t', float64[:]),
        ('charge_max_t', float64[:]),
        ('flexible_max_t', float64[:]),
        ('discharge_max_t_nodal', float64[:]),
        ('charge_max_t_nodal', float64[:]),
        ('flexible_max_t_nodal', float64[:]),
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
    def __init__(self, 
                 static_instance: bool, 
                 node_count: bool,
                 storage_count: bool,
                 flexible_count: bool):
        # PERHAPS REMOVE THIS WHOLE CLASS AND ADD A DYNAMIC INSTANCE OF FLEET
        # WITHOUT ANY DATA LOADED?
        self.static_instance = static_instance

        self.interval = 0

        # Asset values
        self.discharge_max_t = np.zeros(storage_count, dtype=np.float64)
        self.charge_max_t = np.zeros(storage_count, dtype=np.float64)
        self.flexible_max_t = np.zeros(flexible_count, dtype=np.float64)

        # Aggregated nodal values
        self.discharge_max_t_nodal = np.zeros(node_count, dtype=np.float64)
        self.charge_max_t_nodal = np.zeros(node_count, dtype=np.float64)
        self.flexible_max_t_nodal = np.zeros(node_count, dtype=np.float64)
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

    def initialise(self, 
                   t: int,
                   fleet_capacities: FleetCapacities.class_type.instance_type,
                   starting_storage_energy: NDArray[np.float64],
                   starting_flexible_energy: NDArray[np.float64],
                   residual_load_t: NDArray[np.float64],
                   resolution: float,
                   node_count: int,
                   ) -> None:
        self.interval = t
        self.discharge_max_t = np.minimum(
            fleet_capacities.storage_power,
            starting_storage_energy * fleet_capacities.storage_d_efficiencies / resolution
        )
        self.charge_max_t = np.minimum(
            fleet_capacities.storage_power,
            (fleet_capacities.storage_energy - starting_storage_energy) / fleet_capacities.storage_c_efficiencies / resolution
        )
        self.flexible_max_t = np.minimum(
            fleet_capacities.flexible_power,
            starting_flexible_energy / resolution
        )

        self.discharge_max_t_nodal = helpers.aggregate_assets_to_nodes(self.discharge_max_t, fleet_capacities.storage_node_orders, node_count)
        self.charge_max_t_nodal = helpers.aggregate_assets_to_nodes(self.discharge_max_t, fleet_capacities.storage_node_orders, node_count)
        self.flexible_max_t_nodal = helpers.aggregate_assets_to_nodes(self.flexible_max_t, fleet_capacities.flexible_node_orders, node_count)

        self.netload_t = residual_load_t.copy()

        return None

if JIT_ENABLED:
    energybalance_spec = [
        ('static_instance',boolean),
        ('imports', float64[:,:]), 
        ('exports', float64[:,:]), 
        ('residual_load', float64[:,:]), 
        ('deficits', float64[:,:]), 
        ('spillage', float64[:,:]), 

        ('flexible_power', float64[:,:]), 
        ('storage_power', float64[:,:]),
        ('flexible_energy', float64[:,:]), 
        ('storage_energy', float64[:,:]),

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
        
        ####### MOVE ALL THESE TO NODE CLASS??
        self.imports = np.empty((0, 0), dtype=np.float64)
        self.exports = np.empty((0, 0), dtype=np.float64)
        self.residual_load = np.empty((0, 0), dtype=np.float64)
        self.deficits = np.empty((0, 0), dtype=np.float64)
        self.spillage = np.empty((0, 0), dtype=np.float64)

        ## DELETE THESE AND MOVE TO STORAGE CLASS
        self.flexible_power = np.empty((0, 0), dtype=np.float64)
        self.storage_power = np.empty((0, 0), dtype=np.float64)
        self.flexible_energy = np.empty((0, 0), dtype=np.float64)
        self.storage_energy = np.empty((0, 0), dtype=np.float64)
        #########################################

        self.flexible_power_nodal = np.empty((0, 0), dtype=np.float64)
        self.storage_power_nodal = np.empty((0, 0), dtype=np.float64)
        self.flexible_energy_nodal = np.empty((0, 0), dtype=np.float64)
        self.storage_energy_nodal = np.empty((0, 0), dtype=np.float64)

    def create_dynamic_copy(self):
        copy = EnergyBalance(False)
        copy.imports = self.imports.copy()
        copy.exports = self.exports.copy()
        copy.residual_load = self.residual_load.copy()
        copy.deficits = self.deficits.copy()
        copy.spillage = self.spillage.copy()

        copy.flexible_power = self.flexible_power.copy()
        copy.storage_power = self.storage_power.copy()
        copy.flexible_energy = self.flexible_energy.copy()
        copy.storage_energy = self.storage_energy.copy()

        copy.flexible_power_nodal = self.flexible_power_nodal.copy()
        copy.storage_power_nodal = self.storage_power_nodal.copy()
        copy.flexible_energy_nodal = self.flexible_energy_nodal.copy()
        copy.storage_energy_nodal = self.storage_energy_nodal.copy()
        return copy

    def allocate_memory(self, 
                        node_count: int, 
                        intervals_count: int,
                        storage_count: int,
                        flexible_count: int) -> None:
        self.imports = np.zeros((intervals_count, node_count), dtype=np.float64)
        self.exports = np.zeros((intervals_count, node_count), dtype=np.float64)

        self.deficits = np.zeros((intervals_count, node_count), dtype=np.float64)
        self.spillage = np.zeros((intervals_count, node_count), dtype=np.float64)
        
        self.flexible_power = np.zeros((intervals_count, flexible_count), dtype=np.float64)
        self.storage_power = np.zeros((intervals_count, storage_count), dtype=np.float64)
        self.flexible_energy = np.zeros((intervals_count, flexible_count), dtype=np.float64)
        self.storage_energy = np.zeros((intervals_count, storage_count), dtype=np.float64)
        
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
                self.residual_load[interval, node.order] = node.get_data()[interval]

        for generator in generators_typed_dict.values():
            if (generator.data_status == "availability"):
                for interval in range(intervals_count):
                    self.residual_load[interval, generator.node.order] -= generator.get_data("trace")[interval] * generator.capacity
        return None

    def update_residual_load(self, 
                            generators_typed_dict: TypedDict[int64, Generator_InstanceType],
                            intervals_count: int,) -> None:
        if self.static_instance:
            raise_static_modification_error()

        for generator in generators_typed_dict.values():
            if (generator.data.shape[0] > 0) and (generator.new_build > 0):
                for interval in range(intervals_count):
                    self.residual_load[interval, generator.node.order] -= generator.get_data("trace")[interval] * generator.new_build
        return None
    
    def initialise_stored_energy(self, 
                                 storages_typed_dict: TypedDict[int64, Storage_InstanceType]
                                 ) -> None:
        # Initial stored energy held in last index and fetched with t - 1 == 0 - 1 == -1
        for storage in storages_typed_dict.values():
            self.storage_energy_nodal[-1,storage.node.order] = 0.5*storage.energy_capacity # Make it possible to set custom starting SOC later
        return None

    def initialise_flexible_annual_limits(self, 
                                          year,
                                          first_t: int, 
                                          generators_typed_dict: TypedDict[int64, Generator_InstanceType],
                                          ) -> None:
        for generator in generators_typed_dict.values():
            if generator.unit_type == "flexible":
                self.flexible_energy_nodal[first_t-1, generator.node.order] += generator.get_data("annual_constraints_data")[year]
        return None
    
    def check_remaining_deficit(self, t: int) -> bool:
        return sum(self.deficits[t,:]) > 1e-6