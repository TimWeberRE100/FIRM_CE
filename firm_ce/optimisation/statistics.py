import os, re, shutil
import numpy as np
from numpy.typing import NDArray

from firm_ce.common.constants import SAVE_POPULATION
from firm_ce.optimisation.single_time import Solution
from firm_ce.system.components import Fleet_InstanceType
from firm_ce.system.topology import Network_InstanceType
from firm_ce.system.parameters import ScenarioParameters_InstanceType
from firm_ce.io.file_manager import ResultFile
from firm_ce.common.helpers import safe_divide

class Statistics:
    def __init__(self,
                 x_candidate: NDArray[np.float64],
                 parameters_static: ScenarioParameters_InstanceType,
                 fleet_static: Fleet_InstanceType,
                 network_static: Network_InstanceType,
                 solution_results_directory: str,
                 scenario_name: str,
                 balancing_type: str,
                 copy_callback: bool = True):
        self.solution = Solution(x_candidate,
                                parameters_static,
                                fleet_static,
                                network_static,
                                balancing_type) 
        self.solution.evaluate()

        self.results_directory = self.create_solution_directory(solution_results_directory, scenario_name)
        self.copy_temp_files(copy_callback)
        self.result_files = None

    def create_solution_directory(self, result_directory: str, solution_name: str) -> str:
        safe_name = re.sub(r'[^a-zA-Z0-9_\-]', '_', solution_name)
        solution_dir = os.path.join(result_directory, safe_name)
        os.makedirs(solution_dir, exist_ok=True)        
        return solution_dir
    
    def copy_temp_files(self, copy_callback: bool) -> None:
        if copy_callback:
            temp_dir = os.path.join("results", "temp")
            shutil.copy(os.path.join(temp_dir, "callback.csv"), os.path.join(self.results_directory, "callback.csv"))

            if SAVE_POPULATION:
                shutil.copy(os.path.join(temp_dir, "population.csv"), os.path.join(self.results_directory, "population.csv"))
                shutil.copy(os.path.join(temp_dir, "population_energies.csv"), os.path.join(self.results_directory, "population_energies.csv"))
        return None
    
    def generate_result_files(self):
        self.result_files = {
            'capacities': self.generate_capacities_file(),
            'component_costs': self.generate_component_costs_file(),
            #'energy_balance_ASSETS': self.generate_energy_balance_file('assets'),
            'energy_balance_NODES': self.generate_energy_balance_file('nodes'),
            'energy_balance_NETWORK': self.generate_energy_balance_file('network'),            
            'levelised_costs': self.generate_levelised_costs_file(),
            'summary': self.generate_summary_file(),
            'x': self.generate_x_file(),
        }
        return None

    def write_results(self):
        if not self.solution.evaluated:
            print("WARNING: Solution must be evaluated before writing statistics.")
        for result_file in self.result_files.values():
            result_file.write()
        return None
    
    def get_asset_column_count(self, include_minor_lines: bool = True, include_energy_limits: bool = True) -> int:
        return len(self.solution.fleet.generators) + 2*len(self.solution.fleet.storages) + len(self.solution.network.major_lines) \
            + include_minor_lines*len(self.solution.network.minor_lines) + include_energy_limits*self.solution.fleet.count_generator_unit_type('flexible')
    
    def generate_capacities_file(self) -> ResultFile:
        header = []
        data_array = np.empty(self.get_asset_column_count(include_minor_lines=True,include_energy_limits=False), dtype=np.float64)

        column_counter = 0
        for generator in self.solution.fleet.generators.values():
            header.append(generator.name + ' [GW]')
            data_array[column_counter] = generator.capacity
            column_counter += 1

        for storage in self.solution.fleet.storages.values():
            header.append(storage.name + ' [GW]')
            data_array[column_counter] = storage.power_capacity
            column_counter += 1

        for storage in self.solution.fleet.storages.values():
            header.append(storage.name + ' [GWh]')
            data_array[column_counter] = storage.energy_capacity
            column_counter += 1

        for line in self.solution.network.major_lines.values():
            header.append(line.name + ' [GW]')
            data_array[column_counter] = line.capacity
            column_counter += 1

        for line in self.solution.network.minor_lines.values():
            header.append(line.name + ' [GW]')
            data_array[column_counter] = line.capacity
            column_counter += 1

        result_file = ResultFile(
            'capacities', 
            self.results_directory, 
            header, 
            [data_array], 
            decimals=3)
        return result_file  
    
    def generate_component_costs_file(self) -> ResultFile:
        asset_count = len(self.solution.fleet.generators) + len(self.solution.fleet.storages) \
            + len(self.solution.network.major_lines) + len(self.solution.network.minor_lines)
        header = ['Cost Type']
        row_labels = np.array([
            'Annualised Build Cost [$]',
            'Fixed O&M Cost [$]',
            'Variable O&M Cost [$]',
            'Fuel Cost [$]'
        ], dtype=object)
        data_array = np.zeros((4,asset_count))

        col = 0
        for generator in self.solution.fleet.generators.values():
            header.append(generator.name)
            data_array[0, col] = generator.lt_costs.annualised_build
            data_array[1, col] = generator.lt_costs.fom
            data_array[2, col] = generator.lt_costs.vom
            data_array[3, col] = generator.lt_costs.fuel
            col+=1

        for storage in self.solution.fleet.storages.values():
            header.append(storage.name)
            data_array[0, col] = storage.lt_costs.annualised_build
            data_array[1, col] = storage.lt_costs.fom
            data_array[2, col] = storage.lt_costs.vom
            data_array[3, col] = storage.lt_costs.fuel
            col+=1

        for line in self.solution.network.major_lines.values():
            header.append(line.name)
            data_array[0, col] = line.lt_costs.annualised_build
            data_array[1, col] = line.lt_costs.fom
            data_array[2, col] = line.lt_costs.vom
            data_array[3, col] = line.lt_costs.fuel
            col+=1

        for line in self.solution.network.minor_lines.values():
            header.append(line.name)
            data_array[0, col] = line.lt_costs.annualised_build
            data_array[1, col] = line.lt_costs.fom
            data_array[2, col] = line.lt_costs.vom
            data_array[3, col] = line.lt_costs.fuel
            col+=1

        labelled_data_array = np.column_stack((row_labels, data_array))

        result_file = ResultFile(
            'component_costs', 
            self.results_directory, 
            header, 
            labelled_data_array, 
            decimals=None)
        return result_file

    def generate_energy_balance_file(self, aggregation_type: str) -> ResultFile:
        header = []
        match aggregation_type:
            case "assets":
                column_count = 3*len(self.solution.network.nodes) \
                    + self.get_asset_column_count(include_minor_lines=False,include_energy_limits=True)
            case "nodes":
                column_count = 10*len(self.solution.network.nodes) \
                    + len(self.solution.network.major_lines)
            case "network":
                column_count = 10
        data_array = np.zeros((self.solution.static.intervals_count, column_count), dtype=np.float64)

        column_counter = 0
        match aggregation_type:
            case "assets":
                for node in self.solution.network.nodes.values():
                    header.append(node.name + ' Demand [MW]')
                    data_array[:,column_counter] = node.data*1000
                    column_counter += 1

                for generator in self.solution.fleet.generators.values():
                    header.append(generator.name + ' [MW]')
                    match generator.unit_type:
                        case 'flexible':
                            data_array[:,column_counter] = generator.dispatch_power*1000
                        case _:
                            data_array[:,column_counter] = generator.data*generator.capacity*1000
                    column_counter += 1

                for storage in self.solution.fleet.storages.values():
                    header.append(storage.name + ' [MW]')
                    data_array[:,column_counter] = storage.dispatch_power*1000
                    column_counter += 1

                for generator in self.solution.fleet.generators.values():
                    if generator.unit_type == 'flexible':
                        header.append(generator.name + 'Remaining Energy [MWh]')
                        data_array[:,column_counter] = generator.remaining_energy*1000
                        column_counter += 1

                for storage in self.solution.fleet.storages.values():
                    header.append(storage.name + ' Stored Energy [MWh]')
                    data_array[:,column_counter] = storage.stored_energy*1000
                    column_counter += 1

                for node in self.solution.network.nodes.values():
                    header.append(node.name + ' Spillage [MW]')
                    data_array[:,column_counter] = node.spillage*1000
                    column_counter += 1

                for node in self.solution.network.nodes.values():
                    header.append(node.name + ' Deficit [MW]')
                    data_array[:,column_counter] = node.deficits*1000
                    column_counter += 1

                for line in self.solution.network.major_lines.values():
                    header.append(line.name + ' [MW]')
                    data_array[:,column_counter] = line.flows*1000
                    column_counter += 1

            case "nodes":
                for node in self.solution.network.nodes.values():
                    header.append(node.name + ' Demand [MW]')
                    data_array[:,column_counter] = node.data*1000
                    column_counter += 1
                
                for header_item in ['Solar [MW]', 'Wind [MW]', 'Baseload [MW]', 'Flexible Dispatch [MW]', 
                                    'Storage Dispatch [MW]', 'Flexible Remaining [MWh]', 'Stored Energy [MWh]']:
                    for node in self.solution.network.nodes.values():
                        header.append(node.name + f' {header_item}')
                        column_counter += 1
                
                for generator in self.solution.fleet.generators.values():
                    match generator.unit_type:
                        case "solar":
                            column_idx = len(self.solution.network.nodes) + generator.node.order
                            data_array[:,column_idx] += generator.data*generator.capacity*1000
                        case "wind":                            
                            column_idx = 2*len(self.solution.network.nodes) + generator.node.order
                            data_array[:,column_idx] += generator.data*generator.capacity*1000
                        case "baseload":
                            column_idx = 3*len(self.solution.network.nodes) + generator.node.order
                            data_array[:,column_idx] += generator.data*generator.capacity*1000
                        case "flexible":
                            column_idx = 4*len(self.solution.network.nodes) + generator.node.order
                            data_array[:,column_idx] += generator.dispatch_power*1000
                            

                for storage in self.solution.fleet.storages.values():
                    column_idx = 5*len(self.solution.network.nodes) + storage.node.order
                    data_array[:,column_idx] += storage.dispatch_power*1000

                for generator in self.solution.fleet.generators.values():
                    if generator.unit_type == 'flexible':
                        column_idx = 6*len(self.solution.network.nodes) + generator.node.order
                        data_array[:,column_idx] += generator.remaining_energy*1000

                for storage in self.solution.fleet.storages.values():
                    column_idx = 7*len(self.solution.network.nodes) + storage.node.order
                    data_array[:,column_idx] += storage.stored_energy*1000
                for node in self.solution.network.nodes.values():
                    header.append(node.name + ' Spillage [MW]')
                    data_array[:,column_counter] += node.spillage*1000
                    column_counter += 1

                for node in self.solution.network.nodes.values():
                    header.append(node.name + ' Deficit [MW]')
                    data_array[:,column_counter] += node.deficits*1000
                    column_counter += 1

                for line in self.solution.network.major_lines.values():
                    header.append(line.name + ' Flows [MW]')
                    data_array[:,column_counter] += line.flows*1000
                    column_counter += 1

            case "network":
                for header_item in ['Demand [MW]', 
                                    'Solar [MW]', 'Wind [MW]', 'Baseload [MW]', 'Flexible Dispatch [MW]',
                                    'Storage Dispatch [MW]', 'Flexible Remaining [MWh]', 'Stored Energy [MWh]',
                                    'Spillage [MW]', 'Deficit [MW]']:
                    header.append(header_item)
                for node in self.solution.network.nodes.values():
                    data_array[:,0] += node.data*1000
                    data_array[:,8] += node.spillage*1000
                    data_array[:,9] += node.deficits*1000

                for generator in self.solution.fleet.generators.values():
                    match generator.unit_type:
                        case "solar":
                            data_array[:,1] += generator.data*generator.capacity*1000
                        case "wind":
                            data_array[:,2] += generator.data*generator.capacity*1000
                        case "baseload":
                            data_array[:,3] += generator.data*generator.capacity*1000
                        case "flexible":
                            data_array[:,4] += generator.dispatch_power*1000
                            data_array[:,6] += generator.remaining_energy*1000

                for storage in self.solution.fleet.storages.values():
                    data_array[:,5] += storage.dispatch_power*1000
                    data_array[:,7] += storage.stored_energy*1000
        
        result_file = ResultFile(
            f'energy_balance_{aggregation_type.upper()}', 
            self.results_directory, 
            header, 
            data_array, 
            decimals=0)
        return result_file
    
    def generate_levelised_costs_file(self) -> ResultFile:
        header = ['LCOE [$/MWh]', 'LCOG [$/MWh]', 'LCOB [$/MWh]', 
                  'LCOB_storage [$/MWh]', 'LCOB_transmission [$/MWh]', 'LCOB_losses_spillage [$/MWh]']
        data_array = [0., 0., 0., 0., 0., 0.]
        total_energy = np.abs(sum(self.solution.static.year_energy_demand) - self.solution.network.calculate_lt_line_losses()) * 1000

        # LCOE and Total Levelised Values
        for generator in self.solution.fleet.generators.values():
            header.append(generator.name + ' LCOE [$/MWh]')
            if generator.check_unit_type('flexible'):
                data_array.append(generator.lt_costs.get_total() / total_energy)
            else:
                data_array.append(generator.lt_costs.get_total() / total_energy)
            data_array[0] += generator.lt_costs.get_total()
            data_array[1] += generator.lt_costs.get_total()

        for storage in self.solution.fleet.storages.values():
            header.append(storage.name + ' LCOE [$/MWh]')
            data_array.append(storage.lt_costs.get_total() / total_energy)
            data_array[0] += storage.lt_costs.get_total()
            data_array[2] += storage.lt_costs.get_total()
            data_array[3] += storage.lt_costs.get_total()

        for line in self.solution.network.major_lines.values():
            header.append(line.name + ' LCOE [$/MWh]')
            data_array.append(line.lt_costs.get_total() / total_energy)
            data_array[0] += line.lt_costs.get_total()
            data_array[2] += line.lt_costs.get_total()
            data_array[4] += line.lt_costs.get_total()

        for line in self.solution.network.minor_lines.values():
            header.append(line.name + ' LCOE [$/MWh]')
            data_array.append(line.lt_costs.get_total() / total_energy)
            data_array[0] += line.lt_costs.get_total()
            data_array[2] += line.lt_costs.get_total()
            data_array[4] += line.lt_costs.get_total()

        for i in range(5):
            data_array[i] /= total_energy
        data_array[6] = data_array[0] - data_array[1] - data_array[3] - data_array[4]

        # LCOG, LCOS and LCOT
        for generator in self.solution.fleet.generators.values():
            header.append(generator.name + ' LCOG [$/MWh]')
            if generator.check_unit_type('flexible'):
                data_array.append(safe_divide(
                    generator.lt_costs.get_total(), 
                    sum(generator.dispatch_power)*self.solution.static.resolution*1000
                ))
            else:
                data_array.append(safe_divide(
                    generator.lt_costs.get_total(),
                    sum(generator.data*generator.capacity)*self.solution.static.resolution*1000
                ))    
        
        for storage in self.solution.fleet.storages.values():
            header.append(storage.name + ' LCOS [$/MWh_discharged]')
            data_array.append(safe_divide(
                storage.lt_costs.get_total(), 
                sum(np.maximum(storage.dispatch_power,0))*self.solution.static.resolution*1000
            ))

        for line in self.solution.network.major_lines.values():
            header.append(line.name + ' LCOT [$/MWh_flow]')
            data_array.append(safe_divide(
                line.lt_costs.get_total(), 
                sum(np.abs(line.flows))*self.solution.static.resolution*1000
            ))

        for line in self.solution.network.minor_lines.values():
            header.append(line.name + ' LCOT [$/MWh_flow]')
            data_array.append(safe_divide(
                line.lt_costs.get_total(), 
                sum(np.abs(line.flows))*self.solution.static.resolution*1000
            ))

        result_file = ResultFile(
            'levelised_costs', 
            self.results_directory, 
            header, 
            [np.array(data_array, dtype=np.float64)], 
            decimals=2)
        return result_file

    def calculate_annual_average_energy(self, arr: NDArray[np.float64]) -> float:
        return sum(arr) * self.solution.static.resolution / self.solution.static.year_count 

    def generate_summary_file(self) -> ResultFile:
        header = []
        data_array = []

        for node in self.solution.network.nodes.values():
            header.append(node.name + ' Average Annual Demand [GWh]')
            data_array.append(self.calculate_annual_average_energy(node.data))

        for generator in self.solution.fleet.generators.values():
            header.append(generator.name + ' Average Annual Gen [GWh]')
            if generator.check_unit_type('flexible'):
                data_array.append(self.calculate_annual_average_energy(generator.dispatch_power))
            else:
                data_array.append(self.calculate_annual_average_energy(generator.data*generator.capacity))

        for storage in self.solution.fleet.storages.values():
            header.append(storage.name + ' Average Annual Discharge [GWh]')
            data_array.append(self.calculate_annual_average_energy(np.maximum(storage.dispatch_power, 0)))

        for node in self.solution.network.nodes.values():
            header.append(node.name + ' Average Annual Spillage [GWh]')
            data_array.append(self.calculate_annual_average_energy(node.spillage))

        for node in self.solution.network.nodes.values():
            header.append(node.name + ' Average Annual Deficit [GWh]')
            data_array.append(self.calculate_annual_average_energy(node.deficits))

        for line in self.solution.network.major_lines.values():
            header.append(line.name + ' Average Annual Flows [GWh]')
            data_array.append(self.calculate_annual_average_energy(np.abs(line.flows)))
            
        result_file = ResultFile(
            'summary', 
            self.results_directory, 
            header, 
            [np.array(data_array, dtype=np.float64)], 
            decimals=3)
        return result_file
    
    def generate_x_file(self) -> ResultFile:
        result_file = ResultFile(
            'x', 
            self.results_directory, 
            [], 
            [self.solution.x], 
            decimals=None)
        return result_file
    
    def dump(self):
        residual_load_header = [node.name for node in self.solution.network.nodes.values()]
        residual_load_data = np.array([node.residual_load for node in self.solution.network.nodes.values()], dtype=np.float64).T
        residual_load_dump = ResultFile('residual_load',self.results_directory,residual_load_header,residual_load_data).write()
