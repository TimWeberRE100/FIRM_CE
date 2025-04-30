import os
import re
import csv
import numpy as np
from datetime import datetime

from firm_ce.model import Model
from firm_ce.optimisation.solver import Solver
from firm_ce.file_manager import read_initial_guess
from firm_ce.components.costs import calculate_costs


def generate_result_files(result_x, scenario, config):
    dir_path = create_scenario_dir(scenario.name)
    header_gw, header_mw, header_summary, baseload_capacities = get_generator_details(scenario)
    solution = generate_solution(scenario, result_x, config)

    save_capacity_results(dir_path, header_gw, np.hstack((baseload_capacities, result_x)))
    save_interval_results(dir_path, header_mw, solution)
    save_summary_statistics(dir_path, header_summary, solution)
    save_summary_costs(dir_path, solution, scenario)


def create_scenario_dir(scenario_name):
    safe_name = re.sub(r'[^a-zA-Z0-9_\-]', '_', scenario_name)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    dir_path = os.path.join('results', f'{safe_name}_{timestamp}')
    os.makedirs(dir_path, exist_ok=True)
    return dir_path


def generate_solution(scenario, result_x, config):
    return Solver(config, scenario, result_x).statistics()


def get_generator_details(scenario):
    def group_by_type(type_):
        return [g for g in scenario.generators.values() if g.unit_type == type_]

    baseload = group_by_type('baseload')
    solar = group_by_type('solar')
    wind = group_by_type('wind')
    flexible = group_by_type('flexible')

    storages = list(scenario.storages.values())
    lines = list(scenario.lines.values())
    nodes = list(scenario.nodes.values())

    def make_headers(generators, unit):
        return [f"{g.name} [{unit}]" for g in generators]

    header_gw = np.array(
        make_headers(baseload, 'GW') +
        make_headers(solar, 'GW') +
        make_headers(wind, 'GW') +
        make_headers(flexible, 'GW') +
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

    baseload_capacities = np.array([g.capacity for g in baseload])
    return header_gw, header_mw, header_summary, baseload_capacities


def save_csv(path, header, rows, decimals=None):
    with open(path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(header.split(',') if isinstance(header, str) else header)
        for row in rows:
            writer.writerow(np.round(row, decimals=decimals) if decimals is not None else row)
    print(f"Saved to {path}")


def save_capacity_results(dir_path, header, result_x):
    path = os.path.join(dir_path, 'capacities.csv')
    save_csv(path, header, [result_x], decimals=1)


def save_interval_results(dir_path, header, solution):
    interval_array = np.hstack([
        solution.MLoad,
        solution.GBaseload,
        solution.GPV,
        solution.GWind,
        solution.GFlexible,
        solution.SPower,
        solution.Storage,
        -solution.Spillage_nodal,
        solution.Deficit_nodal,
        solution.TFlows
    ]) * 1000

    save_csv(os.path.join(dir_path, 'energy_balance.csv'), header, interval_array, decimals=0)

    network_array = np.vstack([
        solution.MLoad.sum(axis=1),
        solution.GBaseload.sum(axis=1),
        solution.GPV.sum(axis=1),
        solution.GWind.sum(axis=1),
        solution.GFlexible.sum(axis=1),
        solution.SPower.sum(axis=1),
        -solution.Spillage_nodal.sum(axis=1),
        solution.Deficit_nodal.sum(axis=1),
        solution.Storage.sum(axis=1)
    ]) * 1000

    network_header = ['Demand [MW]', 'Baseload [MW]', 'Solar PV [MW]', 'Wind [MW]', 'Flexible [MW]',
                      'Storage Power [MW]', 'Spillage [MW]', 'Deficit [MW]', 'Storage [MWh]']

    save_csv(os.path.join(dir_path, 'energy_balance_NETWORK.csv'), network_header, network_array.T, decimals=0)


def save_summary_statistics(dir_path, header, solution):
    interval_array = np.hstack([
        solution.MLoad.sum(axis=0),
        solution.GBaseload.sum(axis=0),
        solution.GPV.sum(axis=0),
        solution.GWind.sum(axis=0),
        solution.GFlexible.sum(axis=0),
        solution.GDischarge.sum(axis=0),
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


if __name__ == '__main__':
    model = Model()
    result_x = read_initial_guess()
    generate_result_files(result_x, model.scenarios['mekong_imports'], model.config)

    """ for scenario_name in initial_guesses.keys():
        generate_result_files(initial_guesses[scenario_name], model.scenarios[scenario_name]) """
