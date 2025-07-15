from firm_ce.system.components import Generator, Storage, Fuel
from firm_ce.system.costs import UnitCost

def construct_fuel_class():
    return

def construct_generator_class(generator_python, node_names_dict, generator_unit_types_name_dict):
    unitcost_jit = UnitCost(
                        generator_python.cost.capex_p,
                        generator_python.cost.capex_e,
                        generator_python.cost.fom,
                        generator_python.cost.vom,
                        generator_python.cost.lifetime,
                        generator_python.cost.discount_rate,
                        generator_python.cost.fuel_cost_mwh,
                        generator_python.cost.fuel_cost_h,
                        generator_python.cost.transformer_capex,
                    )
    
    return Generator(
        generator_python.id,
        node_names_dict[generator_python.node],
        generator_python.line.id,
        generator_unit_types_name_dict[generator_python.unit_type],
        generator_python.unit_size,   
        generator_python.capacity,    
        unitcost_jit,
    )

def construct_storage_class(storage_python, node_names_dict):
    unitcost_jit = UnitCost(
                        storage_python.cost.capex_p,
                        storage_python.cost.capex_e,
                        storage_python.cost.fom,
                        storage_python.cost.vom,
                        storage_python.cost.lifetime,
                        storage_python.cost.discount_rate,
                        0.0,
                        0.0,
                        storage_python.cost.transformer_capex,
                    )
    
    return Storage(
        storage_python.id,
        node_names_dict[storage_python.node],
        storage_python.line.id,  
        storage_python.power_capacity,  
        storage_python.energy_capacity,    
        storage_python.charge_efficiency,   
        storage_python.discharge_efficiency,   
        unitcost_jit,
    )