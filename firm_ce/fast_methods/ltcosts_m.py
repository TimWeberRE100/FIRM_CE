from firm_ce.system.costs import UnitCost_InstanceType
from firm_ce.common.constants import JIT_ENABLED, FASTMATH

if JIT_ENABLED:
    from numba import njit
else:
    def njit(func=None, **kwargs):
        if func is not None:
            return func
        def wrapper(f):
            return f
        return wrapper

@njit(fastmath=FASTMATH)
def get_total(ltcosts_instance):
    return ltcosts_instance.annualised_build + ltcosts_instance.fom + ltcosts_instance.vom + ltcosts_instance.fuel

@njit(fastmath=FASTMATH)
def get_fixed(ltcosts_instance):
    return ltcosts_instance.vom + ltcosts_instance.fuel

@njit(fastmath=FASTMATH)
def get_variable(ltcosts_instance):
    return ltcosts_instance.annualised_build + ltcosts_instance.fom

@njit(fastmath=FASTMATH)
def get_present_value(ltcosts_instance, discount_rate: float, lifetime: float) -> float:
    return (1-(1+discount_rate)**(-1*lifetime))/discount_rate

@njit(fastmath=FASTMATH)
def calculate_annualised_build(ltcosts_instance,
                                energy_capacity: float,
                                power_capacity: float,
                                line_length: float,
                                unit_costs: UnitCost_InstanceType,
                                year_count: int,
                                asset_type: str,) -> None:
    present_value = get_present_value(ltcosts_instance, unit_costs.discount_rate, unit_costs.lifetime)
    if asset_type == 'generator' or asset_type == 'storage':
        ltcosts_instance.annualised_build = year_count*(energy_capacity * 1e6 * unit_costs.capex_e + power_capacity * 1e6 * unit_costs.capex_p) / present_value if present_value > 1e-6 else 0
    elif asset_type == 'line':
        ltcosts_instance.annualised_build = year_count*(power_capacity * 1e3 * line_length * unit_costs.capex_p + power_capacity * 1e3 * unit_costs.transformer_capex) / present_value if present_value > 1e-6 else 0
    return None

@njit(fastmath=FASTMATH)
def calculate_fom(ltcosts_instance, 
                    power_capacity: float, 
                    years_float: float,
                    line_length: float,
                    unit_costs: UnitCost_InstanceType,
                    asset_type: str) -> None:
    if asset_type == 'generator' or asset_type == 'storage':
        ltcosts_instance.fom = power_capacity * 1e6 * unit_costs.fom * years_float
    elif asset_type == 'line':
        ltcosts_instance.fom = power_capacity * 1e3 * line_length * unit_costs.fom * years_float
    return None

@njit(fastmath=FASTMATH)
def calculate_vom(ltcosts_instance, 
                    generation: float, 
                    unit_costs: UnitCost_InstanceType) -> None:
    ltcosts_instance.vom = generation * 1e3 * unit_costs.vom
    return None

@njit(fastmath=FASTMATH)
def calculate_fuel(ltcosts_instance,
                    generation: float, 
                    unit_hours: float, 
                    unit_costs: UnitCost_InstanceType) -> None:
    ltcosts_instance.fuel = generation * 1e3 * unit_costs.fuel_cost_mwh + unit_hours * unit_costs.fuel_cost_h
    return None