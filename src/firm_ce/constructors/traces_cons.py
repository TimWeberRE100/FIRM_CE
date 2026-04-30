from typing import Dict

import numpy as np
from numpy.typing import NDArray

from firm_ce.fast_methods import generator_m, node_m
from firm_ce.io.file_manager import DataFile
from firm_ce.system.components import Fleet_InstanceType
from firm_ce.system.topology import Network_InstanceType


def filter_leap_days_from_datafile_data(datafile: DataFile) -> None:
    """
    Remove Feb 29 rows from a DataFile's data dict in-place.

    Only acts on files that carry Month and Day columns (demand and generation types).
    Flexible-annual-limit files (Year column only) are left unchanged.

    Parameters:
    -------
    datafile (DataFile): A DataFile instance whose .data dict maps column names to
        NumPy arrays. Modified in-place.

    Returns:
    -------
    None.

    Side-effects:
    -------
    datafile.data is replaced with a new dict whose arrays exclude all rows where
    Month == 2 and Day == 29.
    """
    data = datafile.data
    if "Month" not in data or "Day" not in data:
        return
    mask = ~((data["Month"] == 2) & (data["Day"] == 29))
    datafile.data = {col: arr[mask] for col, arr in data.items()}


def apply_demand_scalars(
    network: Network_InstanceType,
    demand_scalars: NDArray[np.float64],
    year_first_t: NDArray[np.int64],
    intervals_count: int,
) -> None:
    """
    Scale each year's demand and residual_load in-place across all nodes.

    Must be called after load_datafiles_to_network and before
    load_datafiles_to_generators, so that residual_load reflects the scaled
    demand before initial generation is subtracted.

    Parameters:
    -------
    network (Network_InstanceType): A static instance of the Network jitclass whose
        nodes have already had demand loaded.
    demand_scalars (NDArray[np.float64]): Per-year multipliers of length year_count.
        demand_scalars[i] is applied to all intervals of year i.
    year_first_t (NDArray[np.int64]): Array mapping each year index to its first interval.
    intervals_count (int): Total number of intervals in the modelling horizon.

    Returns:
    -------
    None.

    Side-effects:
    -------
    Modifies node.data and node.residual_load in-place for every Node in the network.
    """
    year_count = len(year_first_t)
    for node in network.nodes.values():
        for year_idx in range(year_count):
            t_start = int(year_first_t[year_idx])
            t_end = int(year_first_t[year_idx + 1]) if year_idx < year_count - 1 else intervals_count
            node.data[t_start:t_end] *= demand_scalars[year_idx]
            node.residual_load[t_start:t_end] *= demand_scalars[year_idx]


def select_datafile(
    datafile_type: str,
    object_name: str,
    datafiles_imported_dict: Dict[str, DataFile],
) -> NDArray[np.float64]:
    """
    Locates and returns the a data trace of a specified datafile_type associated with
    either a Generator or Node object based upon the object's name.

    Parameters:
    -------
    datafile_type (str): The type of datafile. Either 'generation', 'flexible_annual_limit',
        or 'demand'.
    object_name (str): The name attribute of the Generator or Node instance.
    datafiles_imported_dict (Dict[str, DataFile]): A dictionary of DataFile instances, where
        the key is a str of the id in `config/datafiles.csv`.

    Returns:
    -------
    NDArray[np.float64]: A 1-dimensional numpy array containing the data trace for the
        specified datafile_type and object_name. If no trace was found, an empty array
        is returned.
    """

    matching_datafiles = [df for df in datafiles_imported_dict.values() if df.type == datafile_type]

    trace = np.empty((0,), dtype=np.float64)
    for datafile in matching_datafiles:
        if object_name in datafile.data.keys():
            trace = np.array(datafile.data[object_name], dtype=np.float64)
            break

    return trace


def load_datafiles_to_generators(
    fleet: Fleet_InstanceType,
    datafiles_imported_dict: Dict[str, DataFile],
    resolution: float,
    year_first_t: NDArray[np.int64],
    intervals_count: int,
) -> None:
    """
    Iterates through all generators in the fleet and loads their time-series data to each
    instance. The baseload, solar, and wind generators are expected to have 'generation'
    traces defining their capacity factor in each time interval, and the flexible generators
    are expected to have a 'flexible_annual_limit' trace defining their maximum generation
    in each year.

    Parameters:
    -------
    fleet (Fleet_InstanceType): A static instance of the Fleet jitclass.
    datafiles_imported_dict (Dict[str, DataFile]): A dictionary of DataFile instances, where
        the key is the id in `config/datafiles.csv`.
    resolution (float): The time resolution of each interval for the input data [hours/interval].
    year_first_t (NDArray[np.int64]): Array mapping each year index to its first time interval index.
    intervals_count (int): Total number of time intervals in the modelling horizon.

    Returns:
    -------
    None.

    Side-effects:
    -------
    The data_status, data, and annual_constraints_data attributes of each generator object are
    modified.

    The residual_load at the node where each generator is located is also updated. The update
    to residual load is based upon the year-specific initial capacity and generation trace. This
    means that load_datafiles_to_network must be run before load_datafiles_to_generators.
    """
    for generator in fleet.generators.values():
        generator_m.load_data(
            generator,
            select_datafile("generation", generator.name, datafiles_imported_dict),
            select_datafile("flexible_annual_limit", generator.name, datafiles_imported_dict),
            resolution,
            year_first_t,
            intervals_count,
        )
    return None


def load_datafiles_to_network(
    network: Network_InstanceType,
    datafiles_imported_dict: Dict[str, DataFile],
) -> None:
    """
    Iterates through all nodes in the network and loads their time-series 'demand' data to each
    instance. The demand data is in units of MW.

    Parameters:
    -------
    network (Network_InstanceType): A static instance of the Network jitclass.
    datafiles_imported_dict (Dict[str, DataFile]): A dictionary of DataFile instances, where
        the key is the id in `config/datafiles.csv`.

    Returns:
    -------
    None.

    Side-effects:
    -------
    The data_status and data attributes of each node object are modified.

    The residual_load at the node where each generator is located is initialised with a copy
    of the demand trace.
    """
    for node in network.nodes.values():
        node_m.load_data(
            node,
            select_datafile("demand", node.name, datafiles_imported_dict)
            / 1000,  # Convert MW to GW - allow custom unit selection in future
        )
    return None


def unload_data_from_generators(fleet: Fleet_InstanceType):
    """
    Iterates through all generators and unloads time-series data. Allows large amounts of
    memory to be cleared before running an optimisation for a new scenario.

    Parameters:
    -------
    fleet (Fleet_InstanceType): A static instance of the Fleet jitclass.

    Returns:
    -------
    None.

    Side-effects:
    -------
    The data_status, data, and annual_constraints_data attributes of each generator object are
    modified.
    """
    for generator in fleet.generators.values():
        generator_m.unload_data(generator)
    return None


def unload_data_from_network(network: Network_InstanceType):
    """
    Iterates through all nodes and unloads time-series data. Allows large amounts of
    memory to be cleared before running an optimisation for a new scenario.

    Parameters:
    -------
    network (Network_InstanceType): A static instance of the Network jitclass.

    Returns:
    -------
    None.

    Side-effects:
    -------
    The data_status and data attributes of each node object are modified.
    """
    for node in network.nodes.values():
        node_m.unload_data(node)
    return None
