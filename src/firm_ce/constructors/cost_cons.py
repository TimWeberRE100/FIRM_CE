from typing import Optional

from ..system.components import Fuel
from ..system.costs import UnitCost, UnitCost_InstanceType


def construct_UnitCost_object(
    capex_p: float,
    fom: float,
    vom: float,
    lifetime: int,
    discount_rate: float,
    heat_rate_base: float = 0.0,
    heat_rate_incr: float = 0.0,
    fuel: Optional[Fuel] = None,
    capex_e: float = 0.0,
    transformer_capex: float = 0.0,
) -> UnitCost_InstanceType:

    fuel_cost_mwh = 0.0
    fuel_cost_h = 0.0
    if fuel:
        fuel_cost_mwh = fuel.cost * heat_rate_incr
        fuel_cost_h = fuel.cost * heat_rate_base

    return UnitCost(
        capex_p,
        fom,
        vom,
        lifetime,
        discount_rate,
        fuel_cost_mwh,
        fuel_cost_h,
        capex_e,
        transformer_capex,
    )
