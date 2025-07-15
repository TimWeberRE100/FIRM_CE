import numpy as np

from firm_ce.common.settings import SETTINGS
from firm_ce.system.energybalance import Traces2d, Fleet

def construct_traces(generators_dict_python, storages_dict_python, intervals):
    generator_unit_types_name_dict = {v : k for (k,v) in SETTINGS['generator_unit_types']}
    
    power_traces_dict = {
        unit_type: np.array([
            gen.generation_trace
            for gen in generators_dict_python.values()
            if gen.unit_type == unit_type
        ], dtype=np.float64, ndmin=2).T
        for unit_type in generator_unit_types_name_dict.keys()
    }

    power_traces_dict['storage'] = np.zeros((intervals, len(storages_dict_python)), dtype=np.float64)

    energy_traces_dict = {
        'storage': np.zeros((intervals, len(storages_dict_python)), dtype=np.float64),
        'flexible': np.zeros((
            intervals,
            sum(1 for gen in generators_dict_python.values() if gen.unit_type == 'flexible')
        ), dtype=np.float64)
    }

    return Traces2d_JIT(
        power_traces_dict['solar'],
        power_traces_dict['wind'],
        power_traces_dict['baseload'],
        power_traces_dict['flexible'],
        power_traces_dict['storage'],
        energy_traces_dict['storage'],
        energy_traces_dict['flexible'],
    )

def construct_generators_dict(generators_dict_python, node_names_dict):
    unit_type_name_to_idx = {v: k for (k, v) in SETTINGS['generator_unit_types']}
    typed_dict = TypedDict.empty(
        key_type=types.int64,
        value_type=Generator_JIT.class_type.instance_type
    )
    for idx, gen in generators_dict_python.items():
        typed_dict[idx] = construct_generator_class(gen, node_names_dict, unit_type_name_to_idx)
    return typed_dict

def construct_storages_dict(storages_dict_python, node_names_dict):
    typed_dict = TypedDict.empty(
        key_type=types.int64,
        value_type=Storage_JIT.class_type.instance_type
    )
    for idx, storage in storages_dict_python.items():
        typed_dict[idx] = construct_storage_class(storage, node_names_dict)
    return typed_dict

def construct_fleet_class(scenario):
    node_names_dict = {scenario.nodes[idx].name : scenario.nodes[idx].id for idx in scenario.nodes}

    generators_dict = construct_generators_dict(scenario.generators, node_names_dict)
    storages_dict = construct_storages_dict(scenario.storages, node_names_dict)

    traces2d = construct_traces(scenario.generators, scenario.storages, scenario.intervals)
    
    return Fleet_JIT(
        generators_dict,
        storages_dict,
        traces2d,
    )