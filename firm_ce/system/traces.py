from numpy.typing import NDArray
import numpy as np

from firm_ce.common.constants import JIT_ENABLED

if JIT_ENABLED:
    from numba.core.types import float64, string
    from numba.experimental import jitclass

    trace_spec = [
        ('status', string),
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
    def __init__(self):
        self.unload_data()

    def allocate_memory(self, flexible_asset_count, storage_asset_count, intervals):
        self.status = "allocated"
        self.flexible_power = np.zeros((intervals,flexible_asset_count), dtype=np.float64)
        self.flexible_remaining_energy = np.zeros((intervals,flexible_asset_count), dtype=np.float64)
        self.storage_power = np.zeros((intervals,storage_asset_count), dtype=np.float64)
        self.storage_remaining_energy = np.zeros((intervals,storage_asset_count), dtype=np.float64)
        return None
    
    def unload_data(self) -> None:
        self.status = "unloaded"
        self.flexible_power = np.empty((0,0), dtype=np.float64)
        self.storage_power = np.empty((0,0), dtype=np.float64)
        self.storage_remaining_energy = np.empty((0,0), dtype=np.float64)
        self.flexible_remaining_energy = np.empty((0,0), dtype=np.float64)
        return None