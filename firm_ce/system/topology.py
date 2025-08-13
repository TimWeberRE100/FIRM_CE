import numpy as np
from numpy.typing import NDArray

from firm_ce.common.exceptions import (
    raise_static_modification_error,
    raise_getting_unloaded_data_error,
)
from firm_ce.common.constants import JIT_ENABLED, NP_FLOAT_MAX
from firm_ce.system.costs import LTCosts, UnitCost_InstanceType, LTCosts_InstanceType

if JIT_ENABLED:
    from numba.core.types import float64, int64, string, boolean, DictType, UniTuple, ListType
    from numba.experimental import jitclass
    from numba.typed.typeddict import Dict as TypedDict
    from numba.typed.typedlist import List as TypedList

    node_spec = [
        ('static_instance',boolean),
        ('id',int64),
        ('order',int64),
        ('name',string),
        ('data_status',string),
        ('data',float64[:]),

        ('residual_load',float64[:]), 

        # Dynamic
        ('storage_merit_order',int64[:]),
        ('flexible_merit_order',int64[:]),

        ('power_capacity',DictType(string,float64)), 
        ('energy_capacity',DictType(string,float64)), 

        ('netload_t',float64),
        ('discharge_max_t',float64[:]),
        ('charge_max_t',float64[:]),
        ('flexible_max_t',float64[:]),

        ('fill',float64),
        ('surplus',float64),
        ('available_imports',float64),

        ('imports',float64[:]),
        ('exports',float64[:]), 
        ('deficits',float64[:]), 
        ('spillage',float64[:]), 

        ('flexible_power',float64[:]), 
        ('storage_power',float64[:]),
        ('flexible_energy',float64[:]), 
        ('storage_energy',float64[:]),         
    ]
else:
    def jitclass(spec):
        def decorator(cls):
            return cls
        return decorator
    node_spec = []

@jitclass(node_spec)
class Node:
    """
    Represents a node (bus) in the network. All nodes require a demand trace
    stored in a datafile within 'data' and referenced in 'config/datafiles.csv'. 
    """

    def __init__(self, static_instance, idx, order, name) -> None:
        """
        Initialize a Node.

        Parameters:
        -------
        id (int): Unique identifier for the node.
        name (str): Name of the node.
        """
        self.static_instance = static_instance
        self.id = idx
        self.order = order # id specific to scenario
        self.name = name
        self.data_status = "unloaded"
        self.data = np.empty((0,), dtype=np.float64)

        self.residual_load = np.empty((0,), dtype=np.float64)
        self.flexible_merit_order = np.empty((0,), dtype=np.int64)
        self.storage_merit_order = np.empty((0,), dtype=np.int64)

        # Dynamic
        self.power_capacity, self.energy_capacity = self.initialise_nodal_capacity()
        
        self.netload_t = 0.0 # GW
        self.discharge_max_t = np.empty((0,), dtype=np.float64) # GW
        self.charge_max_t = np.empty((0,), dtype=np.float64) # GW
        self.flexible_max_t = np.empty((0,), dtype=np.float64) # GW

        self.fill = 0.0 # GW, power attempting to import
        self.surplus = 0.0 # GW, power available for exports
        self.available_imports = 0.0 # GW, maximum power that could be imported from other node surpluses

        self.imports = np.empty((0,), dtype=np.float64)
        self.exports = np.empty((0,), dtype=np.float64)
        self.deficits = np.empty((0,), dtype=np.float64)
        self.spillage = np.empty((0,), dtype=np.float64)

        self.flexible_power = np.empty((0,), dtype=np.float64)
        self.storage_power = np.empty((0,), dtype=np.float64)
        self.flexible_energy = np.empty((0,), dtype=np.float64)
        self.storage_energy = np.empty((0,), dtype=np.float64) 

    def initialise_nodal_capacity(self):
        power_capacity_typed_dict = TypedDict.empty(
            key_type=string,
            value_type=float64
        )
        energy_capacity_typed_dict = TypedDict.empty(
            key_type=string,
            value_type=float64
        )

        power_capacity_typed_dict['solar'] = 0.0
        power_capacity_typed_dict['wind'] = 0.0
        power_capacity_typed_dict['flexible'] = 0.0
        power_capacity_typed_dict['baseload'] = 0.0
        power_capacity_typed_dict['storage'] = 0.0

        energy_capacity_typed_dict['storage'] = 0.0

        return power_capacity_typed_dict, energy_capacity_typed_dict      

    def create_dynamic_copy(self):
        node_copy = Node(
            False,
            self.id,
            self.order,
            self.name
        )
        node_copy.data_status = self.data_status 
        node_copy.data = self.data # This remains static
        node_copy.residual_load = self.residual_load.copy()
        return node_copy        

    def load_data(self, 
                  trace: NDArray[np.float64],):
        self.data_status = "loaded"
        self.data = trace
        self.residual_load = trace.copy()
        return None
    
    def unload_data(self):
        self.data_status = "unloaded"
        self.data = np.empty((0,), dtype=np.float64)
        return None
    
    def get_data(self, data_type):
        if self.data_status == "unloaded":
            raise_getting_unloaded_data_error()

        if data_type == "trace":
            return self.data
        elif data_type == "residual_load":
            return self.residual_load
        else:
            raise RuntimeError("Invalid data_type argument for Node.get_data(data_type).")
        return None
    
    def allocate_memory(self):
        if self.static_instance:
            raise_static_modification_error()
        self.imports = np.zeros_like(self.residual_load, dtype=np.float64)
        self.exports = np.zeros_like(self.residual_load, dtype=np.float64)
        self.deficits = np.zeros_like(self.residual_load, dtype=np.float64)
        self.spillage = np.zeros_like(self.residual_load, dtype=np.float64)

        self.flexible_power = np.zeros_like(self.residual_load, dtype=np.float64)
        self.storage_power = np.zeros_like(self.residual_load, dtype=np.float64)
        self.flexible_energy = np.zeros_like(self.residual_load, dtype=np.float64)
        self.storage_energy = np.zeros_like(self.residual_load, dtype=np.float64)
        return None
    
    def initialise_netload_t(self, interval: int) -> None:
        self.netload_t = self.get_data('residual_load')[interval]
        return None
    
    def update_netload_t(self, interval: int) -> None:
        # Note: exports are negative, so they add to load
        self.netload_t = self.get_data('residual_load')[interval] \
            - self.imports[interval] - self.exports[interval]
        return None
    
    def fill_required(self) -> bool:
        return self.fill > 1e-6
    
    def surplus_available(self) -> bool:
        return self.surplus > 1e-6
    
    def assign_storage_merit_order(self, storages_typed_dict) -> None:
        storages_count = len(storages_typed_dict)
        temp_orders = np.full(storages_count, -1, dtype=np.int64)
        temp_durations = np.full(storages_count, -1, dtype=np.float64)

        idx = 0
        for storage_order, storage in storages_typed_dict.items():
            if storage.node.order == self.order:
                temp_orders[idx] = storage_order
                temp_durations[idx] = storage.duration
                idx += 1

        if idx == 0:
            return

        temp_orders = temp_orders[:idx]
        temp_durations = temp_durations[:idx]

        sort_order = np.argsort(temp_durations)
        self.storage_merit_order = temp_orders[sort_order]
        return None
    
    def assign_flexible_merit_order(self, generators_typed_dict) -> None:
        generators_count = len(generators_typed_dict)
        temp_orders = np.full(generators_count, -1, dtype=np.int64)
        temp_marginal_costs = np.full(generators_count, -1, dtype=np.float64)

        idx = 0
        for generator_order, generator in generators_typed_dict.items():
            if not generator.check_unit_type('flexible'):
                continue

            if generator.node.order == self.order:
                temp_orders[idx] = generator_order
                temp_marginal_costs[idx] = generator.cost.vom + generator.cost.fuel_cost_mwh \
                    + generator.cost.fuel_cost_h * 1000 * generator.unit_size
                idx += 1

        if idx == 0:
            return

        temp_orders = temp_orders[:idx]
        temp_marginal_costs = temp_marginal_costs[:idx]

        sort_order = np.argsort(temp_marginal_costs)
        self.flexible_merit_order = temp_orders[sort_order]
        return None
    
    def check_remaining_netload(self, interval: int, check_case: str) -> bool:
        if check_case == 'deficit':
            return self.netload_t - self.storage_power[interval] - self.flexible_power[interval] > 1e-6
        elif check_case == 'spillage':
            return self.netload_t - self.storage_power[interval] - self.flexible_power[interval] < 1e-6
        elif check_case == 'both':
            return abs(self.netload_t - self.storage_power[interval] - self.flexible_power[interval]) > 1e-6
        return False

Node_InstanceType = Node.class_type.instance_type

if JIT_ENABLED:
    line_spec = [
        ('static_instance',boolean),
        ('id', int64),
        ('order', int64),
        ('name', string),
        ('length', float64),
        ('node_start', Node_InstanceType),
        ('node_end', Node_InstanceType),
        ('loss_factor', float64),
        ('max_build', float64),
        ('min_build', float64),
        ('capacity', float64),
        ('unit_type', string),
        ('near_optimum_check', boolean),
        ('group', string),
        ('cost', UnitCost_InstanceType),

        ('candidate_x_idx',int64),

        # Dynamic
        ('flows',float64[:]),
        ('temp_leg_flows', float64),
        ('lt_flows', float64),

        ('lt_costs',LTCosts_InstanceType),
    ]
else:
    line_spec = []

@jitclass(line_spec)
class Line:
    """
    Represents a transmission line connecting two nodes within the network.

    If node_start and node_end values are empty (np.nan) then the line is a
    minor line that instead connects a generator or storage object to the
    transmission network.
    """
    def __init__(self, 
                 static_instance,
                 idx,
                 order,
                 name,
                 length,
                 node_start,
                 node_end,
                 loss_factor,
                 max_build,
                 min_build,
                 capacity,
                 unit_type,
                 near_optimum_check,
                 group,
                 cost,) -> None:
        """
        Initialize a Line object.
        """
        self.static_instance = static_instance
        self.id = idx
        self.order = order # id specific to scenario
        self.name = name
        self.length = length # km
        self.node_start = node_start # Starting node
        self.node_end = node_end # Ending node
        self.loss_factor = loss_factor # Transmission losses % per 1000 km
        self.max_build = max_build # GW/year
        self.min_build = min_build # GW/year
        self.capacity = capacity # GW
        self.unit_type = unit_type
        self.near_optimum_check = near_optimum_check
        self.group = group
        self.cost = cost

        self.candidate_x_idx = -1

        # Dynamic
        self.flows = np.empty(0, dtype=np.float64) # GW, total line flows
        self.temp_leg_flows = 0.0 # GW, line flows reserved for a route on the current leg
        self.lt_flows = 0.0 # GWh

        self.lt_costs = LTCosts()

    def create_dynamic_copy(self, nodes_typed_dict, line_type):
        if line_type == "major":
            node_start_copy = nodes_typed_dict[self.node_start.order]
            node_end_copy = nodes_typed_dict[self.node_end.order]
        elif line_type == "major":
            node_start_copy = Node(False,-1,-1,"MINOR_NODE")
            node_end_copy = Node(False,-1,-1,"MINOR_NODE")
        
        line_copy = Line(
            False,
            self.id,
            self.order,
            self.name,
            self.length,
            node_start_copy,
            node_end_copy,
            self.loss_factor,
            self.max_build,
            self.min_build,
            self.capacity,
            self.unit_type,
            self.near_optimum_check,
            self.group,
            self.cost, # This remains static
        )
        line_copy.candidate_x_idx = self.candidate_x_idx
        return line_copy

    def check_minor_line(self) -> bool:
        return self.id == -1
    
    def build_capacity(self, new_build_power_capacity):
        if self.static_instance:
            raise_static_modification_error()
        self.capacity += new_build_power_capacity
        return None
    
    def allocate_memory(self, intervals_count: int) -> None:
        if self.static_instance:
            raise_static_modification_error()
        self.flows = np.zeros(intervals_count, dtype=np.float64)
        return None
    
    def calculate_lt_flow(self, resolution: float) -> None:
        self.lt_flows = sum(np.abs(self.flows)) * resolution
        return None
    
    def calculate_lt_costs(self, years_float: float) -> float:
        self.lt_costs.calculate_annualised_build(0.0, self.capacity, 0.0, self.cost, 'line')
        self.lt_costs.calculate_fom(self.capacity, years_float, 0.0, self.cost, 'line')
        self.lt_costs.calculate_vom(self.lt_flows, self.cost)
        self.lt_costs.calculate_fuel(self.lt_flows, 0, self.cost)
        return self.lt_costs.get_total()
    
    def get_lt_losses(self) -> float:
        return self.lt_flows * self.loss_factor * self.length / 1000

Line_InstanceType = Line.class_type.instance_type

if JIT_ENABLED:
    route_spec = [
        ('static_instance',boolean),
        ('initial_node',Node_InstanceType),
        ('nodes',ListType(Node_InstanceType)),
        ('lines',ListType(Line_InstanceType)),   
        ('line_directions',int64[:]),
        ('legs',int64),
        ('flow_update',float64),
    ]
else:
    route_spec = []

@jitclass(route_spec)
class Route:
    def __init__(self,
                 static_instance,
                 initial_node,
                 nodes_typed_list,
                 lines_typed_list,
                 line_directions,
                 legs,):
        self.static_instance = static_instance
        self.initial_node = initial_node
        self.nodes = nodes_typed_list
        self.lines = lines_typed_list
        self.line_directions = line_directions
        self.legs = legs

        # Dynamic
        self.flow_update = 0.0

    def create_dynamic_copy(self,
                            nodes_typed_dict: DictType(int64,Node_InstanceType),
                            lines_typed_dict: DictType(int64,Line_InstanceType)):
        nodes_list_copy = TypedList.empty_list(Node_InstanceType)
        lines_list_copy = TypedList.empty_list(Line_InstanceType)

        for node in self.nodes:
            nodes_list_copy.append(nodes_typed_dict[node.order])

        for line in self.lines:
            lines_list_copy.append(lines_typed_dict[line.order])
        
        return Route(False,
                     nodes_typed_dict[self.initial_node.order],
                     nodes_list_copy,
                     lines_list_copy,
                     self.line_directions.copy(),
                     self.legs)
    
    def check_contains_line(self, new_line: Line_InstanceType) -> bool:
        for line in self.lines:
            if new_line.order == line.order:
                return True
        return False
    
    def check_contains_node(self, new_node: Node_InstanceType) -> bool:
        for node in self.nodes:
            if new_node.order == node.order:
                return True
        return False
    
    def get_max_flow_update(self, interval):
        max_flow = NP_FLOAT_MAX
        for leg in range(self.legs+1):
            max_flow = min(
                max_flow, 
                self.lines[leg].capacity - self.line_directions[leg] * (self.lines[leg].flows[interval] + self.lines[leg].temp_leg_flows)
            )
        return max_flow
    
    def calculate_flow_update(self, interval):
        self.flow_update = min(
            self.nodes[-1].surplus,
            self.get_max_flow_update(interval)
        )
        self.initial_node.available_imports += self.flow_update

        # If multiple routes on the same leg use the same lines, they must be constrained by capacity committed for that leg
        for leg in range(self.legs+1):
            self.lines[leg].temp_leg_flows += self.line_directions[leg] * self.flow_update
        return None
    
    def update_exports(self, interval: int) -> None:
        self.nodes[-1].exports[interval] -= self.flow_update
        self.nodes[-1].surplus -= self.flow_update
        
        for leg in range(self.legs+1):
            self.lines[leg].flows[interval] += self.line_directions[leg] * self.flow_update
        return None

Route_InstanceType = Route.class_type.instance_type
routes_key_type = UniTuple(int64,2)
routes_list_type = ListType(Route_InstanceType)

if JIT_ENABLED:
    network_spec = [
        ('static_instance',boolean),
        ('nodes',DictType(int64,Node_InstanceType)),
        ('major_lines',DictType(int64,Line_InstanceType)),
        ('minor_lines',DictType(int64,Line_InstanceType)),
        ('networksteps_max', int64),
        ('routes',DictType(routes_key_type, routes_list_type)), # Key is Tuple(initial_node.order, legs)
        ('major_line_count',int64),
    ]
else:
    network_spec = []

@jitclass(network_spec)
class Network:
    """
    Constructs the network topology for transmission modeling using lines and nodes.
    Provides access to transmission masks, direct connection matrices, and nth-order networks
    required for transmission business rules in the unit commitment problem.
    """

    def __init__(self, 
                 static_instance,
                 nodes,
                 major_lines,
                 minor_lines,
                 routes,
                 networksteps_max,
                 ) -> None:
        """
        Initialize the Network topology and build all relevant matrices and masks.

        Parameters:
        -------
        lines (Dict[int, Line]): Dictionary of transmission lines.
        nodes (Dict[int, Node]): Dictionary of nodes in the system.
        networksteps_max (int): Maximum number of legs along which transmission can occur.
        """
        self.static_instance = static_instance
        self.nodes = nodes
        self.major_lines = major_lines
        self.minor_lines = minor_lines
        self.routes = routes
        self.networksteps_max = networksteps_max
        self.major_line_count = len(major_lines)
        
    def create_dynamic_copy(self):
        nodes_copy = TypedDict.empty(
            key_type=int64,
            value_type=Node_InstanceType
        )
        major_lines_copy = TypedDict.empty(
            key_type=int64,
            value_type=Line_InstanceType
        )
        minor_lines_copy = TypedDict.empty(
            key_type=int64,
            value_type=Line_InstanceType
        )
        routes_copy = TypedDict.empty(
            key_type=routes_key_type,
            value_type=routes_list_type
        )

        for order, node in self.nodes.items():
            nodes_copy[order] = node.create_dynamic_copy()

        for order, line in self.major_lines.items():
            major_lines_copy[order] = line.create_dynamic_copy(nodes_copy, "major")

        for order, line in self.minor_lines.items():
            minor_lines_copy[order] = line.create_dynamic_copy(nodes_copy, "minor")
        
        for tuple_key, routes_list in self.routes.items():
            routes_copy[tuple_key] = TypedList.empty_list(Route_InstanceType)
            for route in routes_list:
                routes_copy[tuple_key].append(route.create_dynamic_copy(nodes_copy, major_lines_copy))

        network_copy = Network(
            False,
            nodes_copy,
            major_lines_copy,
            minor_lines_copy,
            routes_copy,
            self.networksteps_max,
        )
        network_copy.major_line_count = self.major_line_count
        return network_copy

    def build_capacity(self, decision_x) -> None:
        if self.static_instance:
            raise_static_modification_error()
        for line in self.lines.values():
            line.capacity += decision_x[line.candidate_x_idx]
        return None
    
    def unload_data(self):
        for node in self.nodes.values():
            node.unload_data()
        return None
    
    def allocate_memory(self, intervals_count: int) -> None:
        if self.static_instance:
            raise_static_modification_error()
        for node in self.nodes.values():
            node.allocate_memory()
        for line in self.major_lines.values():
            line.allocate_memory(intervals_count)
        return None
    
    def check_remaining_netloads(self, interval: int, check_case: str) -> bool:
        for node in self.nodes.values():
            if node.check_remaining_netload(interval, check_case):
                return True
        return False
    
    def calculate_period_unserved_power(self, first_t: int, last_t: int):
        unserved_power = 0
        for node in self.nodes.values():
            unserved_power += sum(node.deficits[first_t:last_t+1])
        return unserved_power
    
    def reset_transmission(self, interval: int) -> None:
        for line in self.major_lines.values():
            line.flows[interval] = 0.0
        return None
    
    def reset_flow_updates(self) -> None:
        for route_list in self.routes.values():
            for route in route_list:
                route.flow_update = 0.0
        return None
    
    def check_route_surpluses(self, fill_node: Node_InstanceType, leg: int) -> bool:
        # Check if final node in the route has a surplus available
        for route in self.routes[fill_node.order, leg]:
            if route.nodes[-1].surplus_available():
                return True
        return False
    
    def check_network_surplus(self) -> bool:
        for node in self.nodes.values():
            if node.surplus_available():
                return True
        return False
    
    def check_network_fill(self) -> bool:
        for node in self.nodes.values():
            if node.fill_required():
                return True
        return False
    
    def calculate_node_flow_updates(self, fill_node: Node_InstanceType, leg: int, interval: int) -> None:
        fill_node.available_imports = 0.0
        for route in self.routes[fill_node.order, leg]:
            route.calculate_flow_update(interval)
        return None
    
    def scale_flow_updates_to_fill(self, fill_node: Node_InstanceType, leg: int) -> float:
        if fill_node.available_imports > fill_node.fill:
            scale_factor = fill_node.fill / fill_node.available_imports
            for route in self.routes[fill_node.order, leg]:
                route.flow_update *= scale_factor
        return None
    
    def update_transmission_flows(self, fill_node: Node_InstanceType, leg: int, interval: int) -> None:
        for route in self.routes[fill_node.order, leg]:
            fill_node.imports[interval] += route.flow_update
            fill_node.fill -= route.flow_update
            route.update_exports(interval)
        return None
    
    def update_netloads(self, interval: int) -> None:
        for node in self.nodes.values():
            node.update_netload_t(interval)
        return None
    
    def reset_line_temp_flows(self) -> None:
        for line in self.major_lines.values():
            line.temp_leg_flows = 0.0
        return None
    
    def fill_with_transmitted_surpluses(self, interval) -> None:        
        self.reset_flow_updates() 
        if not (self.check_network_surplus() and self.check_network_fill()):
            return None
        
        for leg in range(self.networksteps_max):
            for node in self.nodes.values():
                if not node.fill_required():
                    continue
                if len(self.routes[node.order, leg]) == 0:
                    continue
                if not self.check_route_surpluses(node, leg):
                    continue
                self.reset_line_temp_flows()
                self.calculate_node_flow_updates(node, leg, interval)
                self.scale_flow_updates_to_fill(node, leg)
                self.update_transmission_flows(node, leg, interval)       
        return None
    
    def set_node_fills_and_surpluses(self, transmission_case: str, interval: int) -> None:
        if transmission_case == 'surplus':
            for node in self.nodes.values():
                node.fill = max(node.netload_t, 0)
                node.surplus = -1*min(node.netload_t, 0)
        elif transmission_case == 'storage_discharge':
            for node in self.nodes.values():
                node.fill = max(node.netload_t - node.storage_power[interval], 0)
                node.surplus = max(node.discharge_max_t[-1] - node.storage_power[interval], 0)
        elif transmission_case == 'flexible':
            for node in self.nodes.values():
                node.fill = max(node.netload_t - node.storage_power[interval] - node.flexible_power[interval], 0)
                node.surplus = max(node.flexible_max_t[-1] - node.flexible_power[interval], 0)
        elif transmission_case == 'storage_charge':
            for node in self.nodes.values():
                node.fill = max(node.charge_max_t[-1] + node.storage_power[interval], 0)
                node.surplus = -min(
                    node.netload_t - min(node.storage_power[interval], 0),
                    0
                    )
        return None
    
    def calculate_spillage_and_deficit(self, interval: int) -> None:
        for node in self.nodes.values():
            node.deficits[interval] = max(node.netload_t - node.storage_power[interval] - node.flexible_power[interval], 0)
            node.spillage[interval] = min(node.netload_t - node.storage_power[interval] - node.flexible_power[interval], 0)
        return None
    
    def assign_storage_merit_orders(self, storages_typed_dict) -> None: # storages_typed_dict: DictType(int64, Storage_InstanceType)
        for node in self.nodes.values():
            node.assign_storage_merit_order(storages_typed_dict)
        return None
    
    def assign_flexible_merit_orders(self, generators_typed_dict) -> None: # generators_typed_dict: DictType(int64, Generators_InstanceType)
        for node in self.nodes.values():
            node.assign_flexible_merit_order(generators_typed_dict)
        return None
    
    def calculate_lt_flows(self, resolution: float) -> None:
        for line in self.major_lines.values():
            line.calculate_lt_flow(resolution)
        return None
    
    def calculate_lt_line_losses(self) -> float:
        total_line_losses = 0.0
        for line in self.major_lines.values():
            total_line_losses += line.get_lt_losses()
        for line in self.minor_lines.values():
            total_line_losses += line.get_lt_losses()
        return total_line_losses
    
Network_InstanceType = Network.class_type.instance_type