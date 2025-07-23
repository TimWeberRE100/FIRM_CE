import numpy as np
from typing import Dict

from firm_ce.system.components import Generator, Storage, Fuel, Fleet
from firm_ce.system.topology import Node, Line
from firm_ce.constructors.cost_cons import construct_UnitCost_object
from firm_ce.constructors.traces_cons import construct_Traces2d_object
from firm_ce.common.constants import JIT_ENABLED

if JIT_ENABLED:
    from numba.typed.typeddict import Dict as TypedDict
    from numba.core.types import DictType, int64

def construct_Fuel_object(fuel_dict: Dict[str,str]) -> Fuel.class_type.instance_type:
    idx = int(fuel_dict['id'])
    name = str(fuel_dict['name'])
    cost = float(fuel_dict['cost'])
    emissions = float(fuel_dict['emissions'])
    return Fuel(idx,
                name,
                cost,
                emissions)

def construct_Generator_object(
        generator_dict: Dict[str,str], 
        fuels_imported_dict: Dict[str, Dict[str, str]],
        nodes_object_dict: TypedDict[int64, Node.class_type.instance_type], 
        lines_object_dict: TypedDict[int64, Line.class_type.instance_type],
        order,
        ) -> Generator.class_type.instance_type:
    idx = int(generator_dict['id'])
    name = str(generator_dict['name'])
    unit_size = float(generator_dict['unit_size']) 
    max_build = float(generator_dict['max_build']) 
    min_build = float(generator_dict['min_build'])  
    capacity = float(generator_dict['initial_capacity']) 
    unit_type = str(generator_dict['unit_type'])
    near_optimum_check = str(generator_dict.get('near_optimum','')).lower() in ('true','1','yes')

    node = next(
        node for node in nodes_object_dict.values()
        if node.name == str(generator_dict['node'])
    )

    fuel_dict = next(
        fuel_dict for fuel_dict in fuels_imported_dict.values()
        if fuel_dict['name'] == str(generator_dict['fuel'])
    )    
    fuel = construct_Fuel_object(fuel_dict)

    line = next(
        line for line in lines_object_dict.values()
        if line.name == str(generator_dict['line'])
    )
        
    raw_group = generator_dict.get('range_group', '')
    if raw_group is None or (isinstance(raw_group, float) and np.isnan(raw_group)) or str(raw_group).strip() == '':
        group = name  
    else:
        group = str(raw_group).strip()
            
    cost = construct_UnitCost_object(
        capex_p=float(generator_dict['capex']),
        fom=float(generator_dict['fom']),
        vom=float(generator_dict['vom']),
        lifetime=int(generator_dict['lifetime']),
        discount_rate=float(generator_dict['discount_rate']),
        heat_rate_base=float(generator_dict['heat_rate_base']), 
        heat_rate_incr=float(generator_dict['heat_rate_incr']), 
        fuel=fuel
    )

    return Generator(
        idx,
        order,
        name,
        unit_size,
        max_build,
        min_build,
        capacity,
        unit_type,
        near_optimum_check,
        node,
        fuel,
        line,
        group,
        cost,
    )

def construct_Storage_object(
        storage_dict: Dict[str, str], 
        nodes_object_dict: TypedDict[int64, Node.class_type.instance_type], 
        lines_object_dict: TypedDict[int64, Line.class_type.instance_type],
        order,
        ) -> Storage.class_type.instance_type:
    idx = int(storage_dict['id'])
    name = str(storage_dict['name'])
    power_capacity = float(storage_dict['initial_power_capacity'])  
    energy_capacity = float(storage_dict['initial_energy_capacity']) 
    duration = int(storage_dict['duration']) if int(storage_dict['duration']) > 0 else 0
    charge_efficiency = float(storage_dict['charge_efficiency']) 
    discharge_efficiency = float(storage_dict['discharge_efficiency'])  
    max_build_p = float(storage_dict['max_build_p'])  
    max_build_e = float(storage_dict['max_build_e'])  
    min_build_p = float(storage_dict['min_build_p']) 
    min_build_e = float(storage_dict['min_build_e'])  
    unit_type = str(storage_dict['unit_type'])
    near_optimum_check = str(storage_dict.get('near_optimum','')).lower() in ('true','1','yes')    
    
    node = next(
        node for node in nodes_object_dict.values()
        if node.name == str(storage_dict['node'])
    )

    line = next(
        line for line in lines_object_dict.values()
        if line.name == str(storage_dict['line'])
    )
        
    raw_group = storage_dict.get('range_group', '')
    if raw_group is None or (isinstance(raw_group, float) and np.isnan(raw_group)) or str(raw_group).strip() == '':
        group = name  
    else:
        group = str(raw_group).strip()
            
    cost = construct_UnitCost_object(
        capex_p=float(storage_dict['capex_p']),
        fom=float(storage_dict['fom']),
        vom=float(storage_dict['vom']),
        lifetime=int(storage_dict['lifetime']),
        discount_rate=float(storage_dict['discount_rate']),
        capex_e=float(storage_dict['capex_e']),
    )
    return Storage(
        idx,
        order,
        name,
        power_capacity,
        energy_capacity,
        duration,
        charge_efficiency,
        discharge_efficiency,
        max_build_p,
        max_build_e,
        min_build_p,
        min_build_e,
        unit_type,
        near_optimum_check,
        node,
        line,
        group,
        cost,
    )

def construct_Fleet_object(
        generators_imported_dict: Dict[str, Dict[str, str]], 
        storages_imported_dict: Dict[str, Dict[str, str]], 
        fuels_imported_dict: Dict[str, Dict[str, str]], 
        lines_object_dict: TypedDict[int64, Line.class_type.instance_type], 
        nodes_object_dict: TypedDict[int64, Node.class_type.instance_type],
    ) -> Fleet.class_type.instance_type:

    generators = TypedDict.empty(
        key_type=int64,
        value_type=Generator.class_type.instance_type
    )
    for order, idx in enumerate(generators_imported_dict):
        generators[order] = construct_Generator_object(
                                generators_imported_dict[idx], 
                                fuels_imported_dict, 
                                nodes_object_dict,
                                lines_object_dict,
                                order,
                            ) 
    
    storages = TypedDict.empty(
        key_type=int64,
        value_type=Storage.class_type.instance_type
    )
    for order, idx in enumerate(storages_imported_dict):
        storages[order] = construct_Storage_object(
                            storages_imported_dict[idx], 
                            nodes_object_dict,
                            lines_object_dict,
                            order,
                        ) 
        
    traces = construct_Traces2d_object()
    return Fleet(generators,
                 storages,
                 traces,)