from typing import Dict

import numpy as np
from numpy.typing import NDArray

from ..fast_methods import generator_m, node_m
from ..io.file_manager import DataFile
from ..system.components import Fleet_InstanceType
from ..system.topology import Network_InstanceType


def select_datafile(
    datafile_type: str,
    generator_name: str,
    datafiles_imported_dict: Dict[str, DataFile],
) -> NDArray[np.float64]:

    matching_datafiles = [df for df in datafiles_imported_dict.values() if df.type == datafile_type]

    trace = np.empty((0,), dtype=np.float64)
    for datafile in matching_datafiles:
        if generator_name in datafile.data.keys():
            trace = np.array(datafile.data[generator_name], dtype=np.float64)
            break

    return trace


def load_datafiles_to_generators(
    fleet: Fleet_InstanceType,
    datafiles_imported_dict: Dict[str, DataFile],
    resolution: float,
) -> None:
    for generator in fleet.generators.values():
        generator_m.load_data(
            generator,
            select_datafile("generation", generator.name, datafiles_imported_dict),
            select_datafile("flexible_annual_limit", generator.name, datafiles_imported_dict),
            resolution,
        )
    return None


def load_datafiles_to_network(
    network: Network_InstanceType,
    datafiles_imported_dict: Dict[str, DataFile],
) -> None:
    for node in network.nodes.values():
        node_m.load_data(
            node,
            select_datafile("demand", node.name, datafiles_imported_dict)
            / 1000,  # Convert MW to GW - allow custom unit selection in future
        )
    return None


def unload_data_from_generators(fleet: Fleet_InstanceType):
    for generator in fleet.generators.values():
        generator_m.unload_data(generator)
    return None


def unload_data_from_network(network: Network_InstanceType):
    for node in network.nodes.values():
        node_m.unload_data(node)
    return None
