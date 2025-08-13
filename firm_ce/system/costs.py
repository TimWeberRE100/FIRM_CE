from firm_ce.common.constants import JIT_ENABLED

if JIT_ENABLED:
    from numba.core.types import float64, int64
    from numba.experimental import jitclass

    unitcost_spec = [
        ('capex_p', float64),
        ('fom', float64),
        ('vom', float64),
        ('lifetime', int64),
        ('discount_rate', float64),
        ('fuel_cost_mwh', float64),
        ('fuel_cost_h', float64),
        ('capex_e', float64),
        ('transformer_capex', float64),
    ]
else:
    def jitclass(spec):
        def decorator(cls):
            return cls
        return decorator
    
    unitcost_spec = []

@jitclass(unitcost_spec)
class UnitCost:
    """
    Represents cost parameters for a generator, storage, or line object.
    """
    def __init__(self, 
                 capex_p,
                 fom,
                 vom,
                 lifetime,
                 discount_rate,
                 fuel_cost_mwh,
                 fuel_cost_h,
                 capex_e,
                 transformer_capex,
                 ) -> None:
        """
        Initialize cost attributes for a Generator, Storage or Line object.

        Parameters:
        -------
        capex_p (float): Power capacity capital cost ($/kW for generator/storage, $/MW-km for line)
        fom (float): Fixed O&M cost ($/kW/year for generator/storage, $/MW/km/year for line)
        vom (float): Variable O&M cost ($/MWh)
        lifetime (int): Asset lifetime in years
        discount_rate (float): Annual discount rate in range [0,1]
        heat_rate_base (float): Constant heat rate term (GJ/h)
        heat_rate_incr (float): First order marginal heat rate term (GJ/MWh)
        fuel (Fuel): Fuel object
        capex_e (float): Energy capacity capital cost ($/kWh for storage only)
        transformer_capex (float): Transformer-specific cost ($/MW)
        length (float): Line length (used for scaling costs and transmission losses)
        """

        self.capex_p = capex_p # $/kW
        self.capex_e = capex_e # $/kWh, non-zero for energy storage
        self.fom = fom # $/kW/year
        self.vom = vom # $/MWh
        self.lifetime = lifetime # years
        self.discount_rate = discount_rate # [0,1]

        self.fuel_cost_mwh = fuel_cost_mwh # $/MWh = $/GJ * GJ/MWh
        self.fuel_cost_h = fuel_cost_h # $/h/unit = $/GJ * GJ/h/unit
        
        self.transformer_capex = transformer_capex # $/kW, non-zero for lines

UnitCost_InstanceType = UnitCost.class_type.instance_type

if JIT_ENABLED:
    ltcosts_spec = [
        ('annualised_build', float64),
        ('fom', float64),
        ('vom', float64),
        ('fuel', float64),
    ]
else:
    ltcosts_spec = []

@jitclass(ltcosts_spec)
class LTCosts:
    def __init__(self):
        self.annualised_build = 0.0
        self.fom = 0.0
        self.vom = 0.0
        self.fuel = 0.0

    def get_total(self):
        return self.annualised_build + self.fom + self.vom + self.fuel

    def get_fixed(self):
        return self.vom + self.fuel
    
    def get_variable(self):
        return self.annualised_build + self.fom
    
    def get_present_value(self, discount_rate: float, lifetime: float) -> float:
        return (1-(1+discount_rate)**(-1*lifetime))/discount_rate
    
    def calculate_annualised_build(self,
                                   energy_capacity: float,
                                   power_capacity: float,
                                   line_length: float,
                                   unit_costs: UnitCost_InstanceType,
                                   asset_type: str) -> None:
        present_value = self.get_present_value(unit_costs.discount_rate, unit_costs.lifetime)
        if asset_type == 'generator' or asset_type == 'storage':
            self.annualised_build = (energy_capacity * 1e6 * unit_costs.capex_e + power_capacity * 1e6 * unit_costs.capex_p) / present_value if present_value > 1e-6 else 0
        elif asset_type == 'line':
            self.annualised_build = (power_capacity * 1e3 * line_length * unit_costs.capex_p + power_capacity * 1e3 * unit_costs.transformer_capex) / present_value if present_value > 1e-6 else 0
        return None
    
    def calculate_fom(self, 
                      power_capacity: float, 
                      years_float: float,
                      line_length: float,
                      unit_costs: UnitCost_InstanceType,
                      asset_type: str) -> None:
        if asset_type == 'generator' or asset_type == 'storage':
            self.fom = power_capacity * 1e6 * unit_costs.fom * years_float
        elif asset_type == 'line':
            self.fom = power_capacity * 1e3 * line_length * unit_costs.fom * years_float
        return None
    
    def calculate_vom(self, 
                      generation: float, 
                      unit_costs: UnitCost_InstanceType) -> None:
        self.vom = generation * 1e3 * unit_costs.vom
        return None
    
    def calculate_fuel(self,
                       generation: float, 
                       unit_hours: float, 
                       unit_costs: UnitCost_InstanceType) -> None:
        self.fuel = generation * 1e3 * unit_costs.fuel_cost_mwh + unit_hours * unit_costs.fuel_cost_h
        return None

LTCosts_InstanceType = LTCosts.class_type.instance_type