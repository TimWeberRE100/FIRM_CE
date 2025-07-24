from typing import Dict, List, Tuple
import numpy as np
from numpy.typing import NDArray

from firm_ce.system.topology import Node, Line, Network
from firm_ce.constructors.cost_cons import construct_UnitCost_object
from firm_ce.io.validate import is_nan
from firm_ce.common.constants import JIT_ENABLED

if JIT_ENABLED:
    from numba.typed.typeddict import Dict as TypedDict
    from numba.core.types import DictType, int64, UniTuple

def construct_Node_object(idx: int, order: int, node_name: str) -> Node.class_type.instance_type:
    return Node(True, idx, order, node_name)

def construct_Line_object(line_dict: Dict[str, str], nodes_object_dict: TypedDict[int64,Node], order: int) -> Line.class_type.instance_type:
    idx = int(line_dict['id'])
    name = str(line_dict['name'])
    length = int(line_dict['length']) 
    loss_factor = float(line_dict['loss_factor'])  
    max_build = float(line_dict['max_build']) 
    min_build = float(line_dict['min_build'])  
    capacity = float(line_dict['initial_capacity']) 
    unit_type = str(line_dict['unit_type'])
    near_opt = str(line_dict.get('near_optimum','')).lower() in ('true','1','yes')
    minor_node = construct_Node_object(-1,-1,"MINOR_NODE")

    raw_group = line_dict.get('range_group','')
    group = (
        name
        if raw_group is None
        or (isinstance(raw_group, float) and np.isnan(raw_group))
        or str(raw_group).strip() == ''
        else str(raw_group).strip()
    )

    node_start = next(
        (node for node in nodes_object_dict.values()
        if not is_nan(line_dict['node_start']) and node.name == str(line_dict['node_start'])),
        minor_node
    )

    node_end = next(
        (node for node in nodes_object_dict.values()
        if not is_nan(line_dict['node_end']) and node.name == str(line_dict['node_end'])),
        minor_node
    )
    
    cost = construct_UnitCost_object(
        float(line_dict['capex']),
        float(line_dict['fom']),
        float(line_dict['vom']),
        int(line_dict['lifetime']),
        float(line_dict['discount_rate']),
        transformer_capex=int(line_dict['transformer_capex']),
    )
    
    return Line(
        True,
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
        near_opt,
        group,
        cost,
    )

def get_topology(major_lines_object_dict: TypedDict[int64,Line],) -> NDArray[np.int64]:
    topology = np.full((len(major_lines_object_dict), 2), -1, dtype=np.int64)
    for order, line in major_lines_object_dict.items():
        topology[order] = [line.node_start.order, line.node_end.order]
    return topology

def build_scenario_network(topology: NDArray[np.int64], node_object_dict: TypedDict[int64, Node]) -> Tuple[NDArray[np.int64], NDArray[np.bool_]]:
    # Not strictly necessary as things are, but will be if nodes are defined as model-level objects 
    # with distinct idx and order down the track
    network_mask = np.array([(topology == order).sum(axis=1).astype(bool) for order in node_object_dict]).sum(axis=0) == 2 
    network = topology[network_mask]
    transmission_mask = np.zeros((len(node_object_dict), network.shape[0]), dtype=bool)
    for line_id, (start, _) in enumerate(network):
        transmission_mask[start, line_id] = True
    return network, transmission_mask

def get_direct_connections(node_count: int, network: NDArray[np.int64]) -> NDArray[np.int64]:
    direct_connections = -1 * np.ones((node_count + 1, node_count + 1), dtype=np.int64)
    for idx, (i, j) in enumerate(network):
        direct_connections[i, j] = idx
        direct_connections[j, i] = idx
    return direct_connections

def network_neighbours(node_idx: int, direct_connections: NDArray[np.int64]) -> NDArray[np.int64]:
    return np.where(direct_connections[node_idx] != -1)[0]

def build_0_donor_cache(nodes_object_dict: TypedDict[int64, Node], direct_connections: NDArray[np.int64]) -> TypedDict[int64, NDArray[np.int64]]:
    cache_0_donors = TypedDict.empty(
        key_type=int64,
        value_type=int64[:, :]
    )
    for order in nodes_object_dict:
        neighboring_node_orders = network_neighbours(order, direct_connections)
        line_orders = direct_connections[order, neighboring_node_orders]
        cache_0_donors[order] = np.stack((neighboring_node_orders, line_orders))
    return cache_0_donors

def extend_route(route: NDArray[np.int64], direct_connections: NDArray[np.int64]) -> list[NDArray[np.int64]]:
    start_neighbors = [node_order for node_order in network_neighbours(route[0], direct_connections) if node_order not in route]
    end_neighbors = [node_order for node_order in network_neighbours(route[-1], direct_connections) if node_order not in route]
    new_routes = []

    for node_order in start_neighbors:
        new_routes.append(np.insert(route, 0, node_order))
    for node_order in end_neighbors:
        new_routes.append(np.append(route, node_order))

    return new_routes

def deduplicate_routes(routes: NDArray[np.int64]) -> NDArray[np.int64]:
    def canonical(row):
        return row if tuple(row) < tuple(row[::-1]) else row[::-1]

    canonical_routes = np.array([canonical(row) for row in routes])
    _, idx = np.unique(canonical_routes, axis=0, return_index=True)
    return canonical_routes[np.sort(idx)]

def nth_order_routes(routes: NDArray[np.int64], direct_connections: NDArray[np.int64]) -> NDArray[np.int64]:
    candidate_routes = []
    for route in routes:
        new_routes = extend_route(route, direct_connections)
        candidate_routes.extend(new_routes)

    deduplicated_routes = deduplicate_routes(np.array(candidate_routes, dtype=np.int64))
    return deduplicated_routes

def build_n_donor_cache(
        network: NDArray[np.int64], 
        networksteps_max: int, 
        nodes_object_dict: TypedDict[int64,Node],
        direct_connections: NDArray[np.int64],
        ) -> Dict[tuple[int, int], NDArray[np.int64]]:
    
    donor_n_cache = TypedDict.empty(
        key_type=UniTuple(int64, 2),
        value_type=int64[:, :, :]
    )
    routes = network.copy()

    for step in range(1, networksteps_max):
        routes = nth_order_routes(routes, direct_connections)
        for node_order in nodes_object_dict:
            forward = routes[routes[:, 0] == node_order]
            reverse = routes[routes[:, -1] == node_order]
            combined = np.vstack((forward[:, 1:], reverse[:, :-1][:, ::-1]))

            line_orders = np.empty_like(combined)
            for i in range(combined.shape[0]):
                line_orders[i, 0] = direct_connections[node_order, combined[i, 0]]
                for j in range(1, combined.shape[1]):
                    line_orders[i, j] = direct_connections[combined[i, j - 1], combined[i, j]]

            donor_n_cache[(node_order, step)] = np.dstack((combined, line_orders)).T
    return donor_n_cache

def construct_Network_object(
        nodes_imported_list: List[str],
        lines_imported_dict: Dict[str, Dict[str,str]],
        networksteps_max: int,
        ) -> Network.class_type.instance_type:
    
    nodes = TypedDict.empty(
        key_type=int64,
        value_type=Node.class_type.instance_type
    )
    for idx in range(len(nodes_imported_list)):
        nodes[idx] = construct_Node_object(idx, idx, nodes_imported_list[idx]) # Separate idx from order in fture version for consistency? 
    
    major_lines = TypedDict.empty(
        key_type=int64,
        value_type=Line.class_type.instance_type
    )
    minor_lines = TypedDict.empty(
        key_type=int64,
        value_type=Line.class_type.instance_type
    )
    order_major = 0
    order_minor = 0
    for idx in lines_imported_dict:
        if not is_nan(lines_imported_dict[idx]['node_start']) and not is_nan(lines_imported_dict[idx]['node_end']):
            major_lines[order_major] = construct_Line_object(lines_imported_dict[idx], nodes, order_major)
            order_major+=1
        else:
            minor_lines[order_minor] = construct_Line_object(lines_imported_dict[idx], nodes, order_minor)
            order_minor+=1
    
    topology = get_topology(major_lines)
    network, transmission_mask = build_scenario_network(topology, nodes)
    direct_connections = get_direct_connections(len(nodes), network)
    cache_0_donors = build_0_donor_cache(nodes, direct_connections)
    cache_n_donors = build_n_donor_cache(network, networksteps_max, nodes, direct_connections)
    transmission_capacities_initial = np.array([line.capacity for line in major_lines.values()], dtype=np.float64)

    return Network(
        True,
        nodes,
        major_lines,
        minor_lines,
        cache_0_donors,
        cache_n_donors,
        transmission_mask,
        networksteps_max,
        transmission_capacities_initial,
    )