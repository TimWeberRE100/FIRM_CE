import os
import re
import csv
import numpy as np
import shutil

from firm_ce.optimisation.solver import Solver
from firm_ce.system.costs import calculate_costs, calculate_cost_components
from firm_ce.common.constants import SAVE_POPULATION

def generate_result_files(result_x, scenario, config, copy_callback=True):
    dir_path = create_scenario_dir(scenario.name, scenario.results_dir)

    if copy_callback:
        temp_dir = os.path.join("results", "temp")
        shutil.copy(os.path.join(temp_dir, "callback.csv"), os.path.join(dir_path, "callback.csv"))

        if SAVE_POPULATION:
            shutil.copy(os.path.join(temp_dir, "population.csv"), os.path.join(dir_path, "population.csv"))
            shutil.copy(os.path.join(temp_dir, "population_energies.csv"), os.path.join(dir_path, "population_energies.csv"))

    header_gw, header_mw, header_summary, header_costs = get_generator_details(scenario)
    solution = generate_solution(scenario, result_x, config)

    save_csv(os.path.join(dir_path, 'x.csv'), result_x, [], decimals=None)

    save_capacity_results(dir_path, header_gw, solution)
    save_interval_results(dir_path, header_mw, solution)
    save_summary_statistics(dir_path, header_summary, solution)
    save_summary_costs(dir_path, solution, scenario)
    save_cost_components(dir_path, solution, scenario, header_costs)


def create_scenario_dir(scenario_name, results_dir):
    safe_name = re.sub(r'[^a-zA-Z0-9_\-]', '_', scenario_name)
    dir_path = os.path.join(results_dir, safe_name)
    os.makedirs(dir_path, exist_ok=True)
    return dir_path


def generate_solution(scenario, result_x, config):
    return Solver(config, scenario).statistics(result_x)


def get_generator_details(scenario):
    def group_by_type(type_):
        return [g for g in scenario.generators.values() if g.unit_type == type_]

    baseload = group_by_type('baseload')
    solar = group_by_type('solar')
    wind = group_by_type('wind')
    flexible = group_by_type('flexible')
    generators = [scenario.generators[idx] for idx in scenario.generators]

    storages = [scenario.storages[idx] for idx in scenario.storages]
    lines = [scenario.lines[idx] for idx in scenario.lines]
    nodes = [scenario.nodes[idx] for idx in scenario.nodes]

    def make_headers(generators, unit=None):
        if unit:
            return [f"{g.name} [{unit}]" for g in generators]
        return [f"{g.name}" for g in generators]

    header_gw = np.array(
        make_headers(generators, 'GW') +
        make_headers(storages, 'GW') +
        [f"{s.name} [GWh]" for s in storages] +
        make_headers(lines, 'GW')
    )

    header_mw = np.array(
        [f"{n.name} Demand [MW]" for n in nodes] +
        make_headers(baseload, 'MW') +
        make_headers(solar, 'MW') +
        make_headers(wind, 'MW') +
        make_headers(flexible, 'MW') +
        make_headers(storages, 'MW') +
        [f"{s.name} [MWh]" for s in storages] +
        [f"{n.name} Spillage [MW]" for n in nodes] +
        [f"{n.name} Deficit [MW]" for n in nodes] +
        make_headers(lines, 'MW')
    )

    header_summary = np.array(
        [f"{n.name} Average Annual Demand [TWh]" for n in nodes] +
        [f"{g.name} Average Annual Gen [TWh]" for g in baseload + solar + wind + flexible] +
        [f"{s.name} Average Annual Discharge [TWh]" for s in storages] +
        [f"{n.name} Average Annual Spillage [TWh]" for n in nodes] +
        [f"{n.name} Average Annual Deficit [TWh]" for n in nodes]
    )

    header_costs = np.array(
        ["Cost Type"] +
        make_headers(generators) +
        make_headers(storages)
    )

    return header_gw, header_mw, header_summary, header_costs


def save_csv(path, header, rows, decimals=None):
    with open(path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(header.split(',') if isinstance(header, str) else header)
        for row in rows:
            writer.writerow(np.round(row, decimals=decimals) if decimals is not None else row)
    print(f"Saved to {path}")


def save_capacity_results(dir_path, header, solution):
    _, _, _, tech_capacities = calculate_costs(solution)
    capacities = np.array([tech_capacities[0][i] for i in range(len(tech_capacities[0])) if tech_capacities[0][i]>0] +
                          [tech_capacities[1][i] for i in range(len(tech_capacities[1])) if tech_capacities[1][i]>0] +
                          [tech_capacities[3][i] for i in range(len(tech_capacities[3])) if tech_capacities[1][i]>0] +
                          [tech_capacities[2][i] for i in range(len(tech_capacities[2])) if tech_capacities[2][i]>0])
    path = os.path.join(dir_path, 'capacities.csv')
    save_csv(path, header, [capacities], decimals=3)

def safe_array(arr, num_rows):
    return arr if arr.size > 0 else np.zeros((num_rows, 0), dtype=np.float64)

def save_interval_results(dir_path, header, solution):
    interval_array = np.hstack([
        solution.MLoad,
        safe_array(solution.GBaseload, solution.intervals),
        safe_array(solution.GPV, solution.intervals),
        safe_array(solution.GWind, solution.intervals),
        safe_array(solution.GFlexible, solution.intervals),
        safe_array(solution.SPower, solution.intervals),
        safe_array(solution.Storage, solution.intervals),
        -solution.Spillage_nodal, 
        solution.Deficit_nodal, 
        safe_array(solution.TFlows, solution.intervals)
    ]) * 1000

    save_csv(os.path.join(dir_path, 'energy_balance.csv'), header, interval_array, decimals=0)

    network_array = np.vstack([
        solution.MLoad.sum(axis=1),
        safe_array(solution.GBaseload, solution.intervals).sum(axis=1),
        safe_array(solution.GPV, solution.intervals).sum(axis=1), 
        safe_array(solution.GWind, solution.intervals).sum(axis=1), 
        safe_array(solution.GFlexible, solution.intervals).sum(axis=1), 
        safe_array(solution.SPower, solution.intervals).sum(axis=1), 
        -solution.Spillage_nodal.sum(axis=1),
        solution.Deficit_nodal.sum(axis=1),
        safe_array(solution.Storage, solution.intervals).sum(axis=1)
    ]) * 1000

    network_header = ['Demand [MW]', 'Baseload [MW]', 'Solar PV [MW]', 'Wind [MW]', 'Flexible [MW]',
                      'Storage Power [MW]', 'Spillage [MW]', 'Deficit [MW]', 'Storage [MWh]']

    save_csv(os.path.join(dir_path, 'energy_balance_NETWORK.csv'), network_header, network_array.T, decimals=0)


def save_summary_statistics(dir_path, header, solution):
    interval_array = np.hstack([
        solution.MLoad.sum(axis=0),
        safe_array(solution.GBaseload, solution.intervals).sum(axis=0), 
        safe_array(solution.GPV, solution.intervals).sum(axis=0), 
        safe_array(solution.GWind, solution.intervals).sum(axis=0), 
        safe_array(solution.GFlexible, solution.intervals).sum(axis=0), 
        safe_array(solution.GDischarge, solution.intervals).sum(axis=0), 
        -solution.Spillage_nodal.sum(axis=0),
        solution.Deficit_nodal.sum(axis=0)
    ]) * solution.resolution / solution.years / 1000

    path = os.path.join(dir_path, 'summary.csv')
    save_csv(path, header, [interval_array], decimals=6)


def save_summary_costs(dir_path, solution, scenario):
    total_cost, tech_costs, tech_annual_gens, tech_capacities = calculate_costs(solution)
    lcoe_denom = (solution.energy - solution.loss) * 1000

    lcoe_total = total_cost / lcoe_denom
    lcoe_tech = np.hstack(tech_costs) / lcoe_denom
    lcog_total = tech_costs[0].sum() / tech_annual_gens[0].sum() / 1000

    lcog_tech_annual_gen = np.array([tech_annual_gens[0][i]*1000 for i in range(len(tech_capacities[0])) if tech_capacities[0][i]>0])
    lcog_tech = []
    for i in range(len(tech_costs[0])):
        if lcog_tech_annual_gen[i]>0:
            lcog_tech.append(tech_costs[0][i] / lcog_tech_annual_gen[i] )
        else:
            lcog_tech.append(0)
    lcog_tech = np.array(lcog_tech)

    lcob_total = lcoe_total - lcog_total
    lcob_storage = tech_costs[1].sum() / lcoe_denom
    lcob_trans = tech_costs[2].sum() / lcoe_denom
    lcob_losses = lcob_total - lcob_storage - lcob_trans

    lcob_storage_tech = tech_costs[1] / lcoe_denom
    lcob_trans_tech = tech_costs[2] / lcoe_denom

    lcos_tech_annual_discharge = np.array([tech_annual_gens[1][i]*1000 for i in range(len(tech_capacities[1])) if tech_capacities[1][i]>0])
    lcos_tech = []
    for i in range(len(tech_costs[1])):
        if lcos_tech_annual_discharge[i]>0:
            lcos_tech.append(tech_costs[1][i] / lcos_tech_annual_discharge[i] )
        else:
            lcos_tech.append(0)
    lcos_tech = np.array(lcos_tech)

    headers = (
        ['LCOE [$/MWh]', 'LCOG [$/MWh]', 'LCOB [$/MWh]', 'LCOB_storage [$/MWh]', 'LCOB_transmission [$/MWh]', 'LCOB_losses_spillage [$/MWh]'] +
        [f"LCOE_{scenario.generators[i].name} [$/MWh]" for i in solution.generator_ids if tech_capacities[0][i] > 0] +
        [f"LCOE_{scenario.storages[i].name} [$/MWh]" for i in solution.storage_ids if tech_capacities[1][i] > 0] +
        [f"LCOE_{scenario.lines[i].name} [$/MWh]" for i in solution.line_ids if tech_capacities[2][i] > 0] +
        [f"LCOG_{scenario.generators[i].name} [$/MWh]" for i in solution.generator_ids if tech_capacities[0][i] > 0] +
        [f"LCOBS_{scenario.storages[i].name} [$/MWh]" for i in solution.storage_ids if tech_capacities[1][i] > 0] +
        [f"LCOBT_{scenario.lines[i].name} [$/MWh]" for i in solution.line_ids if tech_capacities[2][i] > 0] +
        [f"LCOS_{scenario.storages[i].name} [$/MWh]" for i in solution.storage_ids if tech_capacities[1][i] > 0]
    )

    cost_values = np.hstack([
        lcoe_total, lcog_total, lcob_total, lcob_storage, lcob_trans, lcob_losses,
        lcoe_tech, lcog_tech, lcob_storage_tech, lcob_trans_tech, lcos_tech
    ])

    save_csv(os.path.join(dir_path, 'levelised_costs.csv'), headers, [cost_values], decimals=2)

def save_cost_components(dir_path, solution, scenario, header_costs):
    cost_matrix = calculate_cost_components(solution)
    row_labels = ["Annualised Build Cost [$]","Fixed Cost [$]","Variable Cost [$]","Fuel Cost [$]"]
    cost_matrix_with_labels = np.column_stack([row_labels, cost_matrix])

    save_csv(os.path.join(dir_path, 'component_costs.csv'), header_costs, cost_matrix_with_labels, decimals=None)

if __name__ == '__main__':
    from firm_ce.model import Model
    model = Model()
    for scenario in model.scenarios.values():
        scenario.load_datafiles()  
        generate_result_files(scenario.x0, scenario, model.config, False)
        scenario.unload_datafiles()  
