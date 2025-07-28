import numpy as np
from numpy.typing import NDArray

from firm_ce.common.constants import JIT_ENABLED
from firm_ce.system.topology import Network_InstanceType
from firm_ce.system.components import Fleet_InstanceType
from firm_ce.common.helpers import (
    array_min, 
    array_max_2d_axis1, 
    array_sum_2d_axis0, 
    zero_safe_division
)

if JIT_ENABLED:
    from numba import njit
else:
    def njit(func=None, **kwargs):
        if func is not None:
            return func
        def wrapper(f):
            return f
        return wrapper

@njit
#### THIS NEEDS TO BE MOVED OUT OF STATIC INSTANCE
def get_transmission_flows_t(self,
                            Fillt: NDArray[np.float64], 
                            Surplust: NDArray[np.float64], 
                            Importt: NDArray[np.float64], 
                            Exportt: NDArray[np.float64],
                            ) -> NDArray[np.float64]:
        '''Improve efficiency by avioding so many new variable declarations'''
        # The primary connections are simpler (and faster) to model than the general
        #   nthary connection
        # Since many if not most calls of this function only require primary transmission
        #   I have split it out from general nthary transmission to improve speed
        _transmission = np.zeros(len(self.nodes), np.float64)
        leg = 0
        # loop through nodes with deficits
        for n in self.nodes:
            if Fillt[n] < 1e-6:
                continue
            # appropriate slice of network array
            # pdonors is equivalent to donors later on but has different ndim so needs to
            #   be a different variable name for static typing
            pdonors, pdonor_lines = self.cache_0_donors[n]
            _usage = 0.0 # badly named by avoids creating more variables
            for d in pdonors: 
                _usage += Surplust[d]

            if _usage < 1e-6:
                # continue if no surplus to be traded
                continue

            for d, l in zip(pdonors, pdonor_lines):
                _usage = 0.0
                for m in self.nodes:
                    _usage += Importt[m, l]
                # maximum exportable
                _transmission[d] = min(
                    Surplust[d],  # power resource constraint
                    self.transmission_capacities[l] - _usage, # line capacity constraint
                )  

            # scale down to fill requirement
            _usage = 0.0
            for m in self.nodes:
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
                _capacity = np.zeros(self.major_line_count, np.float64)
                # loop through secondary, tertiary, ..., nthary connections
                for leg in range(1, self.networksteps_max):
                    # loop through nodes with deficits
                    for n in self.nodes:
                        if Fillt[n] < 1e-6:
                            continue
                        
                        donors, donor_lines = self.cache_n_donors[(n, leg)]

                        if donors.shape[1] == 0:
                            break  # break if no valid donors
                            
                        _usage = 0.0 # badly named variable but avoids extra variables
                        for d in donors[-1]:
                            _usage += Surplust[d]

                        if _usage < 1e-6:
                            continue

                        _capacity[:] = self.transmission_capacities - array_sum_2d_axis0(Importt)
                        for d, dl in zip(donors[-1], donor_lines.T): # print(d,dl)
                            # power use of each line, clipped to maximum capacity of lowest leg
                            _import[d, dl] = min(array_min(_capacity[dl]), Surplust[d])
                        
                        for l in range(self.major_line_count):
                            # total usage of the line across all import paths
                            _usage=0.0
                            for m in self.nodes:
                                _usage += _import[m, l]
                            # if usage exceeds capacity
                            if _usage > _capacity[l]:
                                # unclear why this raises zero division error from time to time
                                _scale = zero_safe_division(_capacity[l], _usage)
                                for m in self.nodes:
                                    # clip all legs
                                    if _import[m, l] > 1e-6:
                                        for o in self.major_lines:
                                            _import[m, o] *= _scale
                            
                        # intermediate calculation array
                        _transmission = array_max_2d_axis1(_import)
                        
                        # scale down to fill requirement
                        _usage = 0.0
                        for m in self.nodes:
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

        return Importt, Exportt

""" @njit 
def balance_with_neighbouring_surplus(t: int, 
                                      energy_balance: EnergyBalance.class_type.instance_type,) -> bool:
    ### Continue stuff here
    if energy_balance.check_remaining_deficit(t):
        return True
    return False """

@njit 
def initialise_interval(network: Network_InstanceType,
                        fleet: Fleet_InstanceType,
                        interval: int,
                        resolution: float) -> None:
    ####### STILL SLOW WITH MANY FLEXIBLE + STORAGES?
    ####### CAN WE JUST DO EVERYTHING NODALLY INSTEAD?
    ####### APPORTION AT END? HOW TO HANDLE EFFICIENCY?
    ####### ASSUME A MERIT ORDER AT EACH NODE OF STORAGES?
    ####### CREATE LOOKUP TABLE OF CAPACITY + EFFICIENCY VS STORAGE?
    ####### Add a model config option for "faster_balancing" to swap between methods
    for node in network.nodes.values():
        node.netload_t = node.get_data('residual_load')[interval]
    
    for generator in fleet.generators.values():
        if generator.check_unit_type('flexible'):
            generator.set_flexible_max_t(interval, resolution)
    
    for storage in fleet.storages.values():
        storage.set_dispatch_max_t(interval, resolution)
    return None

@njit
def balance_for_period(start_t: int, 
                       end_t: int, 
                       precharging_allowed: bool,
                       solution,
                       ) -> None:
    perform_precharge = False
   
    for t in range(start_t, end_t):
        initialise_interval(
            solution.network,
            solution.fleet,
            t,
            solution.static.resolution
        )

        if not precharging_allowed:
            solution.network.reset_transmission(t)

        """ continue_balancing_check = balance_with_neighbouring_surplus() """
        """ if continue_balancing_check:
            continue_balancing_check = balance_with_local_storage()
        if continue_balancing_check:
            balance_with_neighbouring_storage()
            continue_balancing_check = balance_with_local_flexible()
        if continue_balancing_check:
            transmit_surplus_flag = balance_with_neighbouring_flexible()

        if transmit_surplus_flag:
            transmit_surplus_to_neighbouring_storage()

        energy_balance.update_storage_energy()
        energy_balance.update_flexible_energy()

        if not precharging_allowed:
            continue

        if not perform_precharge and (interval_memory.deficit.sum() > 1e-6):
            perform_precharge = True

        if perform_precharge and (interval_memory.deficit.sum() < 1e-6):
            precharge_storage(t, energy_balance.residual_load)
            perform_precharge = False """

        # Test post-energy balance and during energy-balance methods of apportioning
        # Before building the precharge functionality
    return None