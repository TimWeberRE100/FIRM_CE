import numpy as np

from firm_ce.common.constants import JIT_ENABLED

if JIT_ENABLED:
    from numba import float64, int64, types
    from numba.experimental import jitclass

else:
    def jitclass(spec):
        def decorator(cls):
            return cls
        return decorator
    
    def njit(func=None, **kwargs):
        if func is not None:
            return func
        def wrapper(f):
            return f
        return wrapper

#####################################################
#                       SCENARIO                    #
#####################################################
if JIT_ENABLED:
    scenario_spec = [
        ('intervals', int64),
        ('years', int64),
        ('nodes_count', int64),
        ('lines_count', int64),
        ('resolution', float64),
        ('average_annual_load', float64),
        ('average_annual_trans_loss', float64),
        ('reliability', float64),
        ('fom_scalar', float64),
    ]
else:
    scenario_spec = []

@jitclass(scenario_spec)
class Scenario_JIT:
    def __init__(self,
                 intervals,
                 years,
                 nodes_count,
                 lines_count,
                 resolution,
                 average_annual_load,
                 average_annual_trans_loss,
                 reliability,
                 fom_scalar,):
        self.intervals = intervals
        self.years = years
        self.nodes_count = nodes_count
        self.lines_count = lines_count
        self.resolution = resolution    
        self.average_annual_load = average_annual_load
        self.average_annual_trans_loss = average_annual_trans_loss
        self.reliability = reliability
        self.fom_scalar = fom_scalar # Scale average annual fom to account for leap days for PLEXOS consistency


@jitclass(buses_spec)
class Buses_JIT:
    def __init__(self):
        self.flexible_cap_power
        self.storage_cap_power
        self.storage_cap_energy

        self.power_traces
        self.remaining_energy_traces

        self._charge_max_interval
        self._discharge_max_interval
        self._flexible_max_interval

        self.flexible_sorted_order
        self.storage_sorted_order

        self.imports
        self.exports

""" class Network_JIT:
    def __init__(self):
        self.lines_dict
        self.networksteps        
        self.transmission
        self.cache_0_donors
        self.cache_n_donors

    def get_transmission_flows(self, buses_fill, buses_surplus, buses_imports, buses_exports):
        # The primary connections are simpler (and faster) to model than the general
        #   nthary connection
        # Since many if not most calls of this function only require primary transmission
        #   I have split it out from general nthary transmission to improve speed
        _transmission = np.zeros(solution.nodes, np.float64)
        leg = 0
        # loop through nodes with deficits
        for n in range(solution.nodes):
            if Fillt[n] < 1e-6:
                continue
            # appropriate slice of network array
            # pdonors is equivalent to donors later on but has different ndim so needs to
            #   be a different variable name for static typing
            pdonors, pdonor_lines = solution.cache_0_donors[n]
            _usage = 0.0 # badly named by avoids creating more variables
            for d in pdonors: 
                _usage += Surplust[d]

            if _usage < 1e-6:
                # continue if no surplus to be traded
                continue

            for d, l in zip(pdonors, pdonor_lines):
                _usage = 0.0
                for m in range(solution.nodes):
                    _usage += Importt[m, l]
                # maximum exportable
                _transmission[d] = min(
                    Surplust[d],  # power resource constraint
                    solution.GHvi[l] - _usage, # line capacity constraint
                )  

            # scale down to fill requirement
            _usage = 0.0
            for m in range(solution.nodes):
                _usage += _transmission[m] 
            if _usage > Fillt[n]:
                _scale = Fillt[n] / _usage
                _transmission *= _scale
                _usage *= _scale
            if _usage < 1e-6:
                continue

            # for d, l in zip(pdonors, pdonor_lines):  #  print(d,l)
            for i in range(len(pdonors)):
                # record transmission
                Importt[n, pdonor_lines[i]] += _transmission[pdonors[i]]
                Exportt[pdonors[i], pdonor_lines[i]] -= _transmission[pdonors[i]]
                # adjust deficit/surpluses
                Surplust[pdonors[i]] -= _transmission[pdonors[i]]
                _transmission[pdonors[i]] = 0
    
            Fillt[n] -= _usage

        # Continue with nthary transmission
        # Note: This code block works for primary transmission too, but is slower
        if (Fillt.sum() > 1e-6) and (Surplust.sum() > 1e-6):
                _import = np.zeros(Importt.shape, np.float64)
                _capacity = np.zeros(solution.nhvi, np.float64)
                # loop through secondary, tertiary, ..., nthary connections
                for leg in range(1, solution.networksteps):
                    # loop through nodes with deficits
                    for n in range(solution.nodes):
                        if Fillt[n] < 1e-6:
                            continue
                        
                        donors, donor_lines = solution.cache_n_donors[(n, leg)]

                        if donors.shape[1] == 0:
                            break  # break if no valid donors
                            
                        _usage = 0.0 # badly named variable but avoids extra variables
                        for d in donors[-1]:
                            _usage += Surplust[d]

                        if _usage < 1e-6:
                            continue

                        _capacity[:] = solution.GHvi - array_sum_2d_axis0(Importt)
                        for d, dl in zip(donors[-1], donor_lines.T): # print(d,dl)
                            # power use of each line, clipped to maximum capacity of lowest leg
                            _import[d, dl] = min(array_min(_capacity[dl]), Surplust[d])
                        
                        for l in range(solution.nhvi):
                            # total usage of the line across all import paths
                            _usage=0.0
                            for m in range(solution.nodes):
                                _usage += _import[m, l]
                            # if usage exceeds capacity
                            if _usage > _capacity[l]:
                                # unclear why this raises zero division error from time to time
                                _scale = zero_safe_division(_capacity[l], _usage)
                                for m in range(solution.nodes):
                                    # clip all legs
                                    if _import[m, l] > 1e-6:
                                        for o in range(solution.nhvi):
                                            _import[m, o] *= _scale
                            
                        # intermediate calculation array
                        _transmission = array_max_2d_axis1(_import)
                        
                        # scale down to fill requirement
                        _usage = 0.0
                        for m in range(solution.nodes):
                            _usage += _transmission[m] 
                        if _usage > Fillt[n]:
                            _scale = Fillt[n] / _usage
                            _transmission *= _scale
                            _usage *= _scale
                        if _usage < 1e-6:
                            continue

                        for nd, d, dl in zip(range(donors.shape[1]), donors[-1], donor_lines.T): # print(nd, d, dl)
                            Importt[n, dl[0]] += _transmission[d]
                            Exportt[donors[0, nd], dl[0]] -= _transmission[d]
                            for step in range(leg):
                                Importt[donors[step, nd], dl[step+1]] += _transmission[d]
                                Exportt[donors[step+1, nd], dl[step+1]] -= _transmission[d]

                        # Adjust fill and surplus
                        Fillt[n] -= _usage
                        Surplust -= _transmission
                        
                        _import[:] = 0.0
                        _capacity[:] = 0.0
                        
                        if (Surplust.sum() < 1e-6) or (Fillt.sum() < 1e-6):
                            break

                    if (Surplust.sum() < 1e-6) or (Fillt.sum() < 1e-6):
                        break

        return Importt, Exportt """