from firm_ce.common.constants import JIT_ENABLED
from firm_ce.system.components import Generator, Storage


if JIT_ENABLED:
    from numba.core.types import float64, int64, string, boolean, DictType, UniTuple
    from numba.experimental import jitclass

    trace_spec = [
        ('solar_power', float64[:, :]),
        ('wind_power', float64[:, :]),
        ('baseload_power', float64[:, :]),
        ('flexible_power', float64[:, :]),
        ('storage_power', float64[:, :]),
        ('storage_remaining_energy', float64[:, :]),
        ('flexible_remaining_energy', float64[:, :]),
    ]
    
else:
    def jitclass(spec):
        def decorator(cls):
            return cls
        return decorator
    trace_spec = []

@jitclass(trace_spec)
class Traces2d:
    def __init__(self,
                 solar_power,
                 wind_power,
                 baseload_power,
                 flexible_power,
                 storage_power,
                 storage_remaining_energy,
                 flexible_remaining_energy,):
        self.solar_power = solar_power
        self.wind_power = wind_power
        self.baseload_power = baseload_power
        self.flexible_power = flexible_power
        self.storage_power = storage_power

        self.storage_remaining_energy = storage_remaining_energy
        self.flexible_remaining_energy = flexible_remaining_energy

if JIT_ENABLED:
    interval_memory_spec = []
else:
    interval_memory_spec = []

@jitclass(interval_memory_spec)
class IntervalMemory:
    def __init__(self):
        self.discharge_max_t
        self.charge_max_t
        self.flexible_max_t
        self.netload_t

        # Pre-charging
        self.deficit_block_min
        self.deficit_block_max
        self.storage_reversed_t
        self.flexible_limit_reversed_t

        self.precharge_energy
        self.storage_trickling_reserves
        self.flexible_trickling_reserves
        self.precharge_mask
        self.storage_trickling_mask
        self.flexible_trickling_mask


if JIT_ENABLED:
    energybalance_spec = []
else: 
    energybalance_spec = []

@jitclass(energybalance_spec)
class EnergyBalance:
    def __init__(self,
                 fleet_traces,):
        
        self.interval_memory
        self.fleet_capacities
        self.fleet_traces = fleet_traces
        self.imports
        self.exports
        self.residual_load
        self.flexible_sorted_order
        self.storage_sorted_order


