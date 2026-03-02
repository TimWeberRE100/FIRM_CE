from typing import Any, Dict

import numpy as np

from firm_ce.common.logging import get_logger
from firm_ce.common.typing import DictType, ListType, TypedDict, TypedList, UniTuple, float64, int64
from firm_ce.constructors.cost_cons import construct_UnitCost_object
from firm_ce.fast_methods import route_m
from firm_ce.io.validate import is_nan
from firm_ce.system.topology import (
    Line,
    Line_InstanceType,
    Network,
    Network_InstanceType,
    Node,
    Node_InstanceType,
    Route,
    Route_InstanceType,
)
from firm_ce.system.costs import UnitCost_InstanceType


def construct_Node_object(node_dict: Dict[str, Any], order: int) -> Node_InstanceType:
    """
    Takes data required to initialise a single node object and returns an instance of the Node jitclass.
    The nodes (also called buses) represent a spatial location that is treated as a copper-plate. All
    Generator and Storage assets are located at a particular node in the network.
    Although constructor does not do much at this stage, it has been implemented for future-proofing
    in case nodes become more complex later on.

    Parameters:
    -------
    node_dict (Dict[str, Any]): A dictionary containing the attributes of a single node.
    order (int): The scenario-level id associated with the node.

    Returns:
    -------
    Node_InstanceType: A static instance of the Node jitclass.
    """
    idx = int(node_dict["id"])
    name = str(node_dict["name"])
    return Node(True, idx, order, name)


def construct_Line_object(
    line_id: int,
    year_dict: Dict[int, Dict[str, Any]],
    nodes_object_dict: DictType(int64, Node_InstanceType),
    order: int,
    firstyear: int,
    finalyear: int,
) -> Line_InstanceType:
    """
    Takes data required to initialise a single Line object, builds year-keyed TypedDicts for all
    year-varying attributes, and returns a static instance of the Line jitclass. The Lines define
    interconnections between nodes (buses) in the network.

    node_start, node_end, length, and lifetime are time-invariant fields (read from `lines.csv`).
    For years before the line's first data entry (i.e. the line does not yet exist),
    capacity fields (initial_capacity, max_build, min_build) are set to 0.0 and all other
    year-varying attributes (loss_factor, cost) are copied from the first year with data, so that
    the full topology is always initialised across the entire modelling horizon.

    Parameters:
    -------
    line_id (int): The model-level id for the line.
    year_dict (Dict[int, Dict[str, Any]]): A dictionary of per-year line attribute dicts, keyed
        by year integer. May contain only a subset of [firstyear, finalyear] if the line is added
        partway through the horizon.
    nodes_object_dict (DictType(int64, Node_InstanceType)): A typed dictionary of all Node
        jitclass instances for the scenario, keyed by Node.order.
    order (int): The scenario-specific id for the Line instance.
    firstyear (int): First year of the modelling horizon (inclusive).
    finalyear (int): Final year of the modelling horizon (inclusive).

    Returns:
    -------
    Line_InstanceType: A static instance of the Line jitclass.
    """
    any_year_data = next(iter(year_dict.values()))
    name = str(any_year_data["name"])
    unit_type = str(any_year_data["unit_type"])
    near_opt = str(any_year_data.get("near_optimum", "")).lower() in ("true", "1", "yes")

    raw_group = any_year_data.get("range_group", "")
    group = (
        name
        if raw_group is None or (isinstance(raw_group, float) and np.isnan(raw_group)) or str(raw_group).strip() == ""
        else str(raw_group).strip()
    )

    minor_node = construct_Node_object({"id": -1, "name": "MINOR_NODE"}, -1)
    length = float(any_year_data["length"])
    lifetime = int(any_year_data["lifetime"])
    node_start = next(
        (
            n
            for n in nodes_object_dict.values()
            if not is_nan(any_year_data["node_start"]) and n.name == str(any_year_data["node_start"])
        ),
        minor_node,
    )
    node_end = next(
        (
            n
            for n in nodes_object_dict.values()
            if not is_nan(any_year_data["node_end"]) and n.name == str(any_year_data["node_end"])
        ),
        minor_node,
    )

    loss_factor = TypedDict.empty(key_type=int64, value_type=float64)
    max_build = TypedDict.empty(key_type=int64, value_type=float64)
    min_build = TypedDict.empty(key_type=int64, value_type=float64)
    initial_capacity = TypedDict.empty(key_type=int64, value_type=float64)
    cost = TypedDict.empty(key_type=int64, value_type=UnitCost_InstanceType)

    capacity = 0.0

    # Pre-resolve cost from the first year with data for filling pre-existence years.
    first_yr = year_dict[min(year_dict.keys())]
    first_cost = construct_UnitCost_object(
        float(first_yr["capex"]),
        float(first_yr["fom"]),
        float(first_yr["vom"]),
        lifetime,
        float(first_yr["discount_rate"]),
        transformer_capex=int(first_yr["transformer_capex"]),
    )

    for year_idx, year in enumerate(range(firstyear, finalyear + 1)):
        if year in year_dict:
            yr = year_dict[year]
            loss_factor[year_idx] = float(yr["loss_factor"])
            max_build[year_idx] = float(yr["max_build"])
            min_build[year_idx] = float(yr["min_build"])
            initial_capacity[year_idx] = float(yr["initial_capacity"])

            cost[year_idx] = construct_UnitCost_object(
                float(yr["capex"]),
                float(yr["fom"]),
                float(yr["vom"]),
                lifetime,
                float(yr["discount_rate"]),
                transformer_capex=int(yr["transformer_capex"]),
            )

            if year_idx == 0:
                capacity = float(yr["initial_capacity"])
        else:
            get_logger().debug("Line id %s has no data for year %s; setting capacity fields to 0.", line_id, year)
            loss_factor[year_idx] = float(first_yr["loss_factor"])
            max_build[year_idx] = 0.0
            min_build[year_idx] = 0.0
            initial_capacity[year_idx] = 0.0
            cost[year_idx] = first_cost

    return Line(
        True,
        line_id,
        order,
        name,
        length,
        node_start,
        node_end,
        loss_factor,
        max_build,
        min_build,
        initial_capacity,
        capacity,
        unit_type,
        near_opt,
        group,
        cost,
    )


def construct_new_Route_object(
    initial_node: Node_InstanceType,
    new_node: Node_InstanceType,
    new_line: Line_InstanceType,
    line_direction: int,
    leg: int,
) -> Route_InstanceType:
    """
    Initialises a single route object with the first line in the route. Routes are
    later extended after the Route instance has been created. A Route defines a
    possible sequence of steps through the network.

    Parameters:
    -------
    initial_node (Node_InstanceType): The node where the route begins.
    new_node (Node_InstanceType): The node at the other end of the first line in the route.
    new_line (Line_InstanceType): The first line in the route, connecting the initial_node
        and new_node.
    line_direction (int): Defines the direction of initial_node -> new_node relative to
        line.start_node -> line.end_node. Same direction is +1, different direction is -1.
    leg (int): The total number of steps (i.e., lines) in the route object.

    Returns:
    -------
    Route_InstanceType: A static instance of the Route jitclass.
    """
    route_nodes = TypedList.empty_list(Node_InstanceType)
    route_lines = TypedList.empty_list(Line_InstanceType)

    route_nodes.append(new_node)
    route_lines.append(new_line)

    return Route(True, initial_node, route_nodes, route_lines, np.array([line_direction], dtype=np.int64), leg)


def extend_route(
    route: Route_InstanceType, new_node: Node_InstanceType, new_line: Line_InstanceType, line_direction: int, leg: int
) -> Route_InstanceType:
    """
    Takes an existing route, extends it by a single leg, and returns the extended
    Route instance.

    Parameters:
    -------
    route (Route_InstanceType): An existing Route instance to be extended.
    new_node (Node_InstanceType): The node at the other end of the new_line in the route.
    new_line (Line_InstanceType): The new line in the route, connecting the final node
        in the existing route to the new_node.
    line_direction (int): Defines the direction of route.nodes[-1] -> new_node relative to
        line.start_node -> line.end_node. Same direction is +1, different direction is -1.
    leg (int): The total number of steps (i.e., lines) along the extended route object.

    Returns:
    -------
    Route_InstanceType: A static instance of the Route jitclass.
    """
    route_nodes = route.nodes.copy()
    route_nodes.append(new_node)

    route_lines = route.lines.copy()
    route_lines.append(new_line)

    route_line_directions = list(route.line_directions)
    route_line_directions.append(line_direction)

    return Route(
        True, route.initial_node, route_nodes, route_lines, np.array(route_line_directions, dtype=np.int64), leg
    )


def get_routes_for_node(
    initial_node: Node_InstanceType,
    routes_typed_dict: DictType(UniTuple(int64, 2), ListType(Route_InstanceType)),
    lines_object_dict: DictType(int64, Line_InstanceType),
    leg: int,
) -> ListType(Route_InstanceType):
    """
    Builds a list of routes with length 'leg' to an initial_node. Routes can only use
    each node and line in the network once.
    If no routes have yet been added to the routes_typed_dict for the initial_node,
    the first routes are initialised. Otherwise, the previous leg of each route is
    extended to form the new list.

    Parameters:
    -------
    initial_node (Node_InstanceType): The destination of the route.
    routes_typed_dict (DictType(UniTuple(int64, 2), ListType(Route_InstanceType))):
        A typed dictionary where values are lists of routes to the initial_node. The key
        is a tuple (initial_node.order, leg) so that routes of a specified length to a node
        can quickly be accessed.
    lines_object_dict (DictType(int64, Line_InstanceType)): A typed dictionary of
        all Line jitclass instances for the scenario. Key defined as Line.order.
    leg (int): The total number of steps (i.e., lines) along the route objects for the
        list returned by this function.

    Returns:
    -------
    ListType(Route_InstanceType): A typed list of Route instances. Typed list required for
        JIT-compatibility.
    """
    routes_to_node_curr_leg = TypedList.empty_list(Route_InstanceType)
    key = (initial_node.order, leg - 1)
    if key in routes_typed_dict:
        routes_to_node_prev_leg = routes_typed_dict[key].copy()
        for route in routes_to_node_prev_leg:
            for line in lines_object_dict.values():
                if route_m.check_contains_line(route, line):  # Remove loops
                    continue
                node_start = line.node_start
                node_end = line.node_end
                if node_start is not None and node_start.order == route.nodes[-1].order:
                    if route_m.check_contains_node(route, node_end):  # Remove loops
                        continue
                    new_route = extend_route(route, node_end, line, -1, leg)
                    routes_to_node_curr_leg.append(new_route)
                elif node_end is not None and node_end.order == route.nodes[-1].order:
                    if route_m.check_contains_node(route, node_start):  # Remove loops
                        continue
                    new_route = extend_route(route, node_start, line, 1, leg)
                    routes_to_node_curr_leg.append(new_route)
    else:
        routes_to_node_prev_leg = TypedList.empty_list(Route_InstanceType)
        for line in lines_object_dict.values():
            node_start = line.node_start
            node_end = line.node_end
            if node_start is not None and node_start.order == initial_node.order:
                new_route = construct_new_Route_object(initial_node, node_end, line, -1, leg)
                routes_to_node_curr_leg.append(new_route)
            elif node_end is not None and node_end.order == initial_node.order:
                new_route = construct_new_Route_object(initial_node, node_start, line, 1, leg)
                routes_to_node_curr_leg.append(new_route)
    return routes_to_node_curr_leg


def build_routes_typed_dict(
    networksteps_max: int,
    nodes_object_dict: DictType(int64, Node_InstanceType),
    lines_object_dict: DictType(int64, Line_InstanceType),
) -> DictType(UniTuple(int64, 2), ListType(Route_InstanceType)):
    """
    Builds a typed dictionary of all routes through the network for every node and leg count
    up to networksteps_max. The key is a tuple (node.order, leg) so that routes of a specified
    length to a node can quickly be accessed.

    Parameters:
    -------
    networksteps_max (int): The maximum number of legs allowed in a route for a given scenario.
        Can be adjusted in `config/scenarios.csv`.
    nodes_object_dict (DictType(int64, Node_InstanceType)): A typed dictionary of
        all Node jitclass instances for the scenario. Key defined as Node.order.
    lines_object_dict (DictType(int64, Line_InstanceType)): A typed dictionary of
        all Line jitclass instances for the scenario. Key defined as Line.order.

    Returns:
    -------
    DictType(UniTuple(int64, 2), ListType(Route_InstanceType)): A typed dictionary where values are
        lists of routes to each node. The key is a tuple (node.order, leg) so that routes of a
        specified length to a node can quickly be accessed.
    """
    routes_typed_dict = TypedDict.empty(key_type=UniTuple(int64, 2), value_type=ListType(Route_InstanceType))

    for leg in range(networksteps_max):
        for node in nodes_object_dict.values():
            routes_typed_list = get_routes_for_node(
                node,
                routes_typed_dict,
                lines_object_dict,
                leg,
            )

            routes_typed_dict[(node.order, leg)] = routes_typed_list
    return routes_typed_dict


def construct_Network_object(
    nodes_imported_dict: Dict[int, Dict[str, Any]],
    lines_imported_dict: Dict[int, Dict[int, Dict[str, Any]]],
    networksteps_max: int,
    firstyear: int,
    finalyear: int,
) -> Network_InstanceType:
    """
    Takes data required to initialise a single Network object, casts values into Numba-compatible
    types, builds Node and Line jitclasses, builds transmission routes, and returns an instance of
    the Network jitclass.

    The Network consists of all objects related to network topology and transmission.
    The major_lines are those associated with transmission topology, while minor_lines are those
    required to link Generator and Storage objects to the main transmission network.

    Parameters:
    -------
    nodes_imported_dict (Dict[int, Dict[str, Any]]): A flat dictionary of node attribute dicts,
        grouped by node id. Nodes have no year-varying attributes.
    lines_imported_dict (Dict[int, Dict[int, Dict[str, Any]]]): A year-keyed dictionary of line
        attribute dicts, grouped by line id.
    networksteps_max (int): The maximum number of legs allowed in a route for a given scenario.
        Can be adjusted in `config/scenarios.csv`.
    firstyear (int): First year of the modelling horizon (inclusive).
    finalyear (int): Final year of the modelling horizon (inclusive).

    Returns:
    -------
    Network_InstanceType: A static instance of the Network jitclass.
    """

    nodes = TypedDict.empty(key_type=int64, value_type=Node_InstanceType)
    order_node = 0
    for idx in nodes_imported_dict:
        nodes[order_node] = construct_Node_object(nodes_imported_dict[idx], order_node)
        order_node += 1

    major_lines = TypedDict.empty(key_type=int64, value_type=Line_InstanceType)
    minor_lines = TypedDict.empty(key_type=int64, value_type=Line_InstanceType)
    order_major = 0
    order_minor = 0
    for idx in lines_imported_dict:
        line_year_dict = lines_imported_dict[idx]
        any_year_data = next(iter(line_year_dict.values()))
        if not is_nan(any_year_data["node_start"]) and not is_nan(any_year_data["node_end"]):
            major_lines[order_major] = construct_Line_object(
                idx, line_year_dict, nodes, order_major, firstyear, finalyear
            )
            order_major += 1
        else:
            minor_lines[order_minor] = construct_Line_object(
                idx, line_year_dict, nodes, order_minor, firstyear, finalyear
            )
            order_minor += 1

    routes = build_routes_typed_dict(networksteps_max, nodes, major_lines)

    return Network(True, nodes, major_lines, minor_lines, routes, networksteps_max)
