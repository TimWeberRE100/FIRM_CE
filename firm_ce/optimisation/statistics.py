import re
from datetime import datetime
import os
import numpy as np
import csv

from firm_ce.model import Model
from firm_ce.optimisation.solver import Solver
from firm_ce.file_manager import read_initial_guess

def generate_result_files(result_x, scenario, config):
    dir_path = create_scenario_dir(scenario.name)
    header_gw, header_mw, header_summary, header_costs, baseload_capacities = get_generator_details(scenario)
    solution = generate_solution(scenario, result_x, config)

    save_capacity_results(dir_path, header_gw, np.hstack((baseload_capacities,result_x)))
    save_interval_results(dir_path, header_mw, solution)
    save_summary_statistics(dir_path, header_summary, solution)
    save_summary_costs(dir_path, header_costs, solution)
    return

def create_scenario_dir(scenario_name):
    sanitised_name = re.sub(r'[^a-zA-Z0-9_\-]', '_', scenario_name)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    dir_path = f'results/{sanitised_name}_{timestamp}' 
    os.makedirs(dir_path, exist_ok=True)
    
    return dir_path

def generate_solution(scenario, result_x, config):
    solver = Solver(config, scenario, result_x)
    solution = solver.statistics()
    return solution

def get_generator_details(scenario):
    baseload_generators = [scenario.generators[idx] for idx in scenario.generators if scenario.generators[idx].unit_type == 'baseload']
    solar_generators = [scenario.generators[idx] for idx in scenario.generators if scenario.generators[idx].unit_type == 'solar']
    wind_generators = [scenario.generators[idx] for idx in scenario.generators if scenario.generators[idx].unit_type == 'wind']
    flexible_generators = [scenario.generators[idx] for idx in scenario.generators if scenario.generators[idx].unit_type == 'flexible']
    storages = [scenario.storages[idx] for idx in scenario.storages]
    lines = [scenario.lines[idx] for idx in scenario.lines]
    nodes = [scenario.nodes[idx] for idx in scenario.nodes]

    baseload_header = [generator.name + ' [GW]' for generator in baseload_generators]
    solar_header = [generator.name + ' [GW]' for generator in solar_generators]
    wind_header = [generator.name + ' [GW]' for generator in wind_generators]
    flexible_p_header = [generator.name + ' [GW]' for generator in flexible_generators]
    storage_p_header = [storage.name + ' [GW]' for storage in storages] 
    storage_e_header = [storage.name + ' [GWh]' for storage in storages] 
    line_header = [line.name + ' [GW]' for line in lines]
    header_gw = np.array(baseload_header + solar_header + wind_header + flexible_p_header + storage_p_header + storage_e_header + line_header)

    demand_header = [node.name + ' Demand [MW]' for node in nodes]
    baseload_header = [generator.name + ' [MW]' for generator in baseload_generators]
    solar_header = [generator.name + ' [MW]' for generator in solar_generators]
    wind_header = [generator.name + ' [MW]' for generator in wind_generators]
    flexible_p_header = [generator.name + ' [MW]' for generator in flexible_generators]
    storage_p_header = [storage.name + ' [MW]' for storage in storages] 
    storage_e_header = [storage.name + ' [MWh]' for storage in storages] 
    spillage_header = [node.name + ' Spillage [MW]' for node in nodes]
    deficit_header = [node.name + ' Deficit [MW]' for node in nodes]
    line_header = [line.name + ' [MW]' for line in lines]
    header_mw = np.array(demand_header + baseload_header + solar_header + wind_header + flexible_p_header + storage_p_header + storage_e_header + spillage_header + deficit_header + line_header)

    demand_header = [node.name + ' Average Annual Demand [TWh]' for node in nodes]
    baseload_header = [generator.name + ' Average Annual Gen [TWh]' for generator in baseload_generators]
    solar_header = [generator.name + ' Average Annual Gen [TWh]' for generator in solar_generators]
    wind_header = [generator.name + ' Average Annual Gen [TWh]' for generator in wind_generators]
    flexible_p_header = [generator.name + ' Average Annual Gen [TWh]' for generator in flexible_generators]
    storage_p_header = [storage.name + ' Average Annual Discharge [TWh]' for storage in storages]
    spillage_header = [node.name + ' Average Annual Spillage [TWh]' for node in nodes]
    deficit_header = [node.name + ' Average Annual Deficit [TWh]' for node in nodes]
    header_summary = np.array(demand_header + baseload_header + solar_header + wind_header + flexible_p_header + storage_p_header + spillage_header + deficit_header)

    levelised_header = ['Total LCOE [$/MWh]','Total LCOG [$/MWh]','Total LCOB [$/MWh]']
    baseload_header = [generator.name + ' LCOG [$/MWh]' for generator in baseload_generators]
    solar_header = [generator.name + ' LCOG [$/MWh]' for generator in solar_generators]
    wind_header = [generator.name + ' LCOG [$/MWh]' for generator in wind_generators]
    flexible_p_header = [generator.name + ' LCOG [$/MWh]' for generator in flexible_generators]
    storage_p_header = [storage.name + ' LCOB [$/MWh]' for storage in storages]
    line_header = [line.name + ' LCOB [$/MWh]' for line in lines]
    spillage_header = [node.name + ' Spillage/Losses LCOB [$/MWh]' for node in nodes]
    header_costs = np.array([levelised_header + baseload_header + solar_header + wind_header + flexible_p_header + storage_p_header + line_header + spillage_header])

    baseload_capacities = np.array([generator.capacity for generator in baseload_generators])
    return header_gw, header_mw, header_summary, header_costs, baseload_capacities

def save_capacity_results(dir_path, header, result_x):
    csv_path = os.path.join(dir_path, 'capacities.csv')

    with open(csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)

        if isinstance(header, str):
            writer.writerow(header.split(','))
        else:
            writer.writerow(header)

        writer.writerow(np.round(result_x, decimals=1))

    print(f"Capacities saved to {csv_path}")
    return

def save_interval_results(dir_path, header, solution):
    interval_array = np.hstack((solution.MLoad,solution.GBaseload,solution.GPV,solution.GWind,solution.GFlexible,solution.SPower,solution.Storage,-solution.Spillage_nodal,solution.Deficit_nodal,solution.TFlows))*1000
    csv_path = os.path.join(dir_path, 'energy_balance.csv')

    with open(csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)

        if isinstance(header, str):
            writer.writerow(header.split(','))
        else:
            writer.writerow(header)

        for row in np.round(interval_array, decimals=0):
            writer.writerow(row)

    interval_array = np.vstack((solution.MLoad.sum(axis=1),solution.GBaseload.sum(axis=1),solution.GPV.sum(axis=1),solution.GWind.sum(axis=1),solution.GFlexible.sum(axis=1),solution.SPower.sum(axis=1),-solution.Spillage_nodal.sum(axis=1),solution.Deficit_nodal.sum(axis=1),solution.Storage.sum(axis=1)))*1000
    network_header = np.array(['Demand [MW]','Baseload [MW]','Solar PV [MW]','Wind [MW]','Flexible [MW]','Storage Power [MW]','Spillage [MW]','Deficit [MW]','Storage [MWh]'])
    csv_path = os.path.join(dir_path, 'energy_balance_NETWORK.csv')
    
    with open(csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)

        if isinstance(network_header, str):
            writer.writerow(network_header.split(','))
        else:
            writer.writerow(network_header)

        for row in np.round(interval_array.T, decimals=0):
            writer.writerow(row)

    print(f"Energy balance saved to {csv_path}")
    return

def save_summary_statistics(dir_path, header, solution):
    interval_array = np.hstack((solution.MLoad.sum(axis=0),solution.GBaseload.sum(axis=0),solution.GPV.sum(axis=0),solution.GWind.sum(axis=0),solution.GFlexible.sum(axis=0),solution.GDischarge.sum(axis=0),-solution.Spillage_nodal.sum(axis=0),solution.Deficit_nodal.sum(axis=0)))*solution.resolution/solution.years/1000
    csv_path = os.path.join(dir_path, 'summary.csv')

    with open(csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)

        if isinstance(header, str):
            writer.writerow(header.split(','))
        else:
            writer.writerow(header)

        writer.writerow(np.round(interval_array,decimals=6))
    return

def save_summary_costs(dir_path, header, solution):
    
    return

if __name__ == '__main__':
    model = Model()
    result_x = read_initial_guess()
    generate_result_files(result_x, model.scenarios['mekong_imports'], model.config)

    """ for scenario_name in initial_guesses.keys():
        generate_result_files(initial_guesses[scenario_name], model.scenarios[scenario_name]) """
