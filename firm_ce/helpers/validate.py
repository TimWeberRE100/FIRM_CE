import numpy as np

from firm_ce.helpers.file_manager import import_csv_data
from firm_ce.helpers.helpers import parse_comma_separated

class ModelData:
    def __init__(self) -> None:
        objects = import_csv_data()

        self.scenarios = objects.get('scenarios')
        self.generators = objects.get('generators')
        self.fuels = objects.get('fuels')
        self.lines = objects.get('lines')
        self.storages = objects.get('storages')
        self.config = objects.get('config')
        self.x0s = objects.get('initial_guess')
        self.settings = objects.get('settings')

    def validate(self):
        return validate_config(self)
    
# Check config.csv
# Check datafiles.csv
# Check fuels.csv
# Check generators.csv
# Check initial_guess.csv
# Check lines.csv
# Check scenarios.csv
# Check storages.csv
# Check each generation datafile
# Check electricity.csv
# Check annual_limits.csv

def validate_model_config(config_dict):
    flag = True
    for item in config_dict.values():
        if item['name'] == 'mutation':
            if not ((float(item['value'])) >= 0 and (float(item['value']) < 2)):
                print("'mutation' must be float in range [0,2)")
                flag = False
        
        elif item['name'] == 'iterations':
            if not (int(item['value']) > 0):
                print("'iterations' must be integer greater than 0")
                flag = False

        elif item['name'] == 'population':
            if not (int(item['value']) > 0):
                print("'population' multiplier must be integer greater than 0")
                flag = False

        elif item['name'] == 'recombination':
            if not ((float(item['value'])) >= 0 and (float(item['value']) <= 1)):
                print("'recombination' must be float in range [0,1]")
                flag = False

        elif item['name'] == 'type':
            if not ((item['value'] == 'single_time') or (item['value'] == 'capacity_expansion')):
                print("'type' must be string with value from ['single_time','capacity_expansion']")
                flag = False

        else:
            print(f"Unknown configuration name '{item['name']}'")

    return flag  

def validate_scenarios(scenarios_dict):
    flag = True
    scenarios_list = []
    scenario_nodes = {}
    scenario_lines = {}

    for item in scenarios_dict.values():
        if item['scenario_name'] in scenarios_list:
            print(f"Duplicate scenario name '{item['value']}'")
            flag = False
        scenarios_list.append(item['scenario_name'])

        if not (float(item['resolution']) > 0):
            print("'resolution' must be float greater than 0")
            flag = False

        if not ((float(item['allowance'])) >= 0 and (float(item['allowance']) <= 1)):
            print("'allowance' must be float in range [0,1]")
            flag = False

        firstyear = int(item['firstyear'])

        finalyear = int(item['finalyear'])

        scenario_nodes[item['scenario_name']] = parse_comma_separated(item['nodes'])

        scenario_lines[item['scenario_name']] = parse_comma_separated(item['lines'])
        
    if firstyear > finalyear:
        print("'firstyear' must be less than 'finalyear'")
        flag = False

    return scenarios_list, scenario_nodes, scenario_lines, flag

def validate_fuels(fuels_dict, scenarios_list):
    flag = True
    scenario_fuels = {scenario : [] for scenario in scenarios_list}

    for idx in fuels_dict:
        if float(fuels_dict[idx]['emissions']) < 0:
            print("'emissions' must be float greater than or equal to 0")
            flag = False

        if float(fuels_dict[idx]['cost']) < 0:
            print("'cost' must be float greater than or equal to 0")
            flag = False

        for scenario in parse_comma_separated(fuels_dict[idx]['scenarios']):
            if scenario in scenarios_list:
                scenario_fuels[scenario].append(fuels_dict[idx]['name'])
            else:
                print(f"WARNING: 'scenario' {scenario} for fuel.id {idx} not defined in scenarios.csv")            

    return scenario_fuels, flag

def validate_lines(lines_dict, scenarios_list, scenario_nodes):
    flag = True
    scenario_lines = {scenario: [] for scenario in scenarios_list}
    scenario_minor_lines = {scenario: [] for scenario in scenarios_list}

    for idx in lines_dict:
        if int(lines_dict[idx]['length']) < 0:
            print("'length' (kilometres) must be integer greater than or equal to 0")
            flag = False
        
        if float(lines_dict[idx]['capex']) < 0:
            print("'capex' ($/kW/km) must be float greater than or equal to 0")
            flag = False

        if float(lines_dict[idx]['transformer_capex']) < 0:
            print("'transformer_capex' ($/kW/km) must be float greater than or equal to 0")
            flag = False

        if float(lines_dict[idx]['fom']) < 0:
            print("'fom' ($/kW/year) must be float greater than or equal to 0")
            flag = False

        if float(lines_dict[idx]['vom']) < 0:
            print("'vom' ($/kWh) must be float greater than or equal to 0")
            flag = False

        if int(lines_dict[idx]['lifetime']) < 0:
            print("'lifetime' (years) must be integer greater than or equal to 0")
            flag = False

        if (float(lines_dict[idx]['discount_rate']) < 0) or (float(lines_dict[idx]['discount_rate']) > 1):
            print("'discount_rate' (%/year) must be float in range [0,1]")
            flag = False

        if (float(lines_dict[idx]['loss_factor']) < 0) or (float(lines_dict[idx]['loss_factor']) >= 1):
            print("'loss_factor' (%/1000km) must be float in range [0,1)")
            flag = False

        if float(lines_dict[idx]['initial_capacity']) < 0:
            print("'initial_capacity' (GW) must be integer greater than or equal to 0")
            flag = False

        if float(lines_dict[idx]['max_build']) < 0:
            print("'max_build' (GW) must be integer greater than or equal to 0")
            flag = False

        if float(lines_dict[idx]['min_build']) < 0:
            print("'min_build' (GW) must be integer greater than or equal to 0")
            flag = False

        if float(lines_dict[idx]['min_build']) > float(lines_dict[idx]['max_build']) :
            print("'min_build' (GW) must be less than 'max_build' (GW)")
            flag = False

        for scenario in parse_comma_separated(lines_dict[idx]['scenarios']):
            if scenario in scenarios_list:
                scenario_lines[scenario].append(lines_dict[idx]['name'])

                if ((lines_dict[idx]['node_start'] not in scenario_nodes[scenario]) and not (isinstance(lines_dict[idx]['node_start'], float) and np.isnan(lines_dict[idx]['node_start']))):
                    print(f"'node_start' {lines_dict[idx]['node_start']} for line {lines_dict[idx]['name']} is not defined in scenario {scenario}")
                    flag = False

                if ((lines_dict[idx]['node_end'] not in scenario_nodes[scenario])  and not (isinstance(lines_dict[idx]['node_end'], float) and np.isnan(lines_dict[idx]['node_end']))):
                    print(f"'node_end' {lines_dict[idx]['node_end']} for line {lines_dict[idx]['name']} is not defined in scenario {scenario}")
                    flag = False

                if ((isinstance(lines_dict[idx]['node_start'], float) and np.isnan(lines_dict[idx]['node_start'])) or (isinstance(lines_dict[idx]['node_end'], float) and np.isnan(lines_dict[idx]['node_end']))):
                    scenario_minor_lines[scenario].append(lines_dict[idx]['name'])

            else:
                print(f"WARNING: 'scenario' {scenario} for line.id {idx} not defined in scenarios.csv")   

    return scenario_lines, scenario_minor_lines, flag

def validate_generators(generators_dict, scenarios_list, scenario_fuels, scenario_lines, scenario_nodes):
    flag = True
    scenario_generators = {scenario: [] for scenario in scenarios_list}
    scenario_baseload = {scenario: [] for scenario in scenarios_list}

    for idx in generators_dict:
        if float(generators_dict[idx]['capex']) < 0:
            print("'capex' ($/kW) must be float greater than or equal to 0")
            flag = False

        if float(generators_dict[idx]['fom']) < 0:
            print("'fom' ($/kW/year) must be float greater than or equal to 0")
            flag = False

        if float(generators_dict[idx]['vom']) < 0:
            print("'fom' ($/kWh) must be float greater than or equal to 0")
            flag = False

        if int(generators_dict[idx]['lifetime']) < 0:
            print("'lifetime' (years) must be integer greater than or equal to 0")
            flag = False

        if (float(generators_dict[idx]['discount_rate']) < 0) or (float(generators_dict[idx]['discount_rate']) > 1):
            print("'discount_rate' (%/year) must be float in range [0,1]")
            flag = False

        if float(generators_dict[idx]['heat_rate_base']) < 0:
            print("'heat_rate_base' (GJ/h) must be float greater than or equal to 0")
            flag = False

        if float(generators_dict[idx]['heat_rate_incr']) < 0:
            print("'heat_rate_incr' (GJ/MWh) must be float greater than or equal to 0")
            flag = False

        if float(generators_dict[idx]['initial_capacity']) < 0:
            print("'initial_capacity' (GW) must be integer greater than or equal to 0")
            flag = False

        if float(generators_dict[idx]['max_build']) < 0:
            print("'max_build' (GW) must be integer greater than or equal to 0")
            flag = False

        if float(generators_dict[idx]['min_build']) < 0:
            print("'min_build' (GW) must be integer greater than or equal to 0")
            flag = False

        if float(generators_dict[idx]['min_build']) > float(generators_dict[idx]['max_build']) :
            print("'min_build' (GW) must be less than 'max_build' (GW)")
            flag = False

        if generators_dict[idx]['unit_type'] not in ['solar','wind','flexible','baseload']:
            print("'unit_type' must be string with value from ['solar','wind','flexible','baseload']")
            flag = False

        for scenario in parse_comma_separated(generators_dict[idx]['scenarios']):
            if scenario in scenarios_list:
                if generators_dict[idx]['name'] in scenario_generators[scenario]:
                    print(f"Duplicate generator name '{generators_dict[idx]['name']}' for scenario {scenario}")
                    flag = False
                else:
                    scenario_generators[scenario].append(generators_dict[idx]['name'])

                if generators_dict[idx]['unit_type'] == 'baseload':
                    scenario_baseload[scenario].append(generators_dict[idx]['name'])

                if (generators_dict[idx]['node'] not in scenario_nodes[scenario]):
                    print(f"'node' {generators_dict[idx]['node']} for generator {generators_dict[idx]['name']} is not defined in scenario {scenario}")
                    flag = False

                if (generators_dict[idx]['fuel'] not in scenario_fuels[scenario]):
                    print(f"'fuel' {generators_dict[idx]['fuel']} for generator {generators_dict[idx]['name']} is not defined in scenario {scenario}")
                    flag = False

                if (generators_dict[idx]['line'] not in scenario_lines[scenario]):
                    print(f"'line' {generators_dict[idx]['line']} for generator {generators_dict[idx]['name']} is not defined in scenario {scenario}")
                    flag = False
                
            else:
                print(f"WARNING: 'scenario' {scenario} for generator.id {idx} not defined in scenarios.csv") 

    return scenario_generators, scenario_baseload, flag

def validate_storages(storages_dict, scenarios_list, scenario_nodes, scenario_lines):
    flag = True
    scenario_storages = {scenario: [] for scenario in scenarios_list}

    for idx in storages_dict:
        if float(storages_dict[idx]['capex_p']) < 0:
            print("'capex_p' ($/kW) must be float greater than or equal to 0")
            flag = False

        if float(storages_dict[idx]['capex_e']) < 0:
            print("'capex_e' ($/kWh) must be float greater than or equal to 0")
            flag = False

        if float(storages_dict[idx]['fom']) < 0:
            print("'fom' ($/kW/year) must be float greater than or equal to 0")
            flag = False

        if float(storages_dict[idx]['vom']) < 0:
            print("'fom' ($/kWh) must be float greater than or equal to 0")
            flag = False

        if int(storages_dict[idx]['lifetime']) < 0:
            print("'lifetime' (years) must be integer greater than or equal to 0")
            flag = False

        if int(storages_dict[idx]['duration']) < 0:
            print("'duration' (hours) must be integer greater than or equal to 0. If value is 0 h, then energy capacity will be used")
            flag = False

        if (float(storages_dict[idx]['discount_rate']) < 0) or (float(storages_dict[idx]['discount_rate']) > 1):
            print("'discount_rate' (%/year) must be float in range [0,1]")
            flag = False

        if (float(storages_dict[idx]['charge_efficiency']) < 0) or (float(storages_dict[idx]['charge_efficiency']) > 1):
            print("'charge_efficiency' (%) must be float in range [0,1]")
            flag = False

        if (float(storages_dict[idx]['discharge_efficiency']) < 0) or (float(storages_dict[idx]['discharge_efficiency']) > 1):
            print("'discharge_efficiency' (%) must be float in range [0,1]")
            flag = False

        if float(storages_dict[idx]['initial_power_capacity']) < 0:
            print("'initial_power_capacity' (GW) must be integer greater than or equal to 0")
            flag = False

        if float(storages_dict[idx]['initial_energy_capacity']) < 0:
            print("'initial_energy_capacity' (GWh) must be integer greater than or equal to 0")
            flag = False

        if float(storages_dict[idx]['max_build_p']) < 0:
            print("'max_build_p' (GW) must be integer greater than or equal to 0")
            flag = False

        if float(storages_dict[idx]['min_build_p']) < 0:
            print("'max_build_p' (GW) must be integer greater than or equal to 0")
            flag = False

        if float(storages_dict[idx]['max_build_e']) < 0:
            print("'max_build_e' (GWh) must be integer greater than or equal to 0")
            flag = False

        if float(storages_dict[idx]['min_build_e']) < 0:
            print("'min_build_e' (GWh) must be integer greater than or equal to 0")
            flag = False

        if float(storages_dict[idx]['min_build_p']) > float(storages_dict[idx]['max_build_p']) :
            print("'min_build_p' (GW) must be less than 'max_build_p' (GW)")
            flag = False

        if float(storages_dict[idx]['min_build_e']) > float(storages_dict[idx]['max_build_e']) :
            print("'min_build_e' (GWh) must be less than 'max_build_e' (GWh)")
            flag = False

        for scenario in parse_comma_separated(storages_dict[idx]['scenarios']):
            if scenario in scenarios_list:
                if storages_dict[idx]['name'] in scenario_storages[scenario]:
                    print(f"Duplicate generator name '{storages_dict[idx]['name']}' for scenario {scenario}")
                    flag = False
                else:
                    scenario_storages[scenario].append(storages_dict[idx]['name'])

                if (storages_dict[idx]['node'] not in scenario_nodes[scenario]):
                    print(f"'node' {storages_dict[idx]['node']} for generator {storages_dict[idx]['name']} is not defined in scenario {scenario}")
                    flag = False

                if (storages_dict[idx]['line'] not in scenario_lines[scenario]):
                    print(f"'line' {storages_dict[idx]['line']} for generator {storages_dict[idx]['name']} is not defined in scenario {scenario}")
                    flag = False
                
            else:
                print(f"WARNING: 'scenario' {scenario} for storage.id {idx} not defined in scenarios.csv") 
    
    return scenario_storages, flag

def validate_initial_guess(x0s_dict, scenarios_list, scenario_generators, scenario_storages, scenario_lines, scenario_baseload, scenario_minor_lines):
    flag = True
    initial_guess_scenarios = []

    for idx in x0s_dict:
        if x0s_dict[idx]['scenario'] not in scenarios_list:
            print(f"WARNING: 'scenario' {x0s_dict[idx]['scenario']} in initial_guess.csv not defined in scenarios.csv") 

        initial_guess_scenarios.append(x0s_dict[idx]['scenario'])

        x0 = parse_comma_separated(x0s_dict[idx]['x_0'])
        bound_length = len(
                            scenario_generators[x0s_dict[idx]['scenario']] 
                            + scenario_storages[x0s_dict[idx]['scenario']] 
                            + scenario_storages[x0s_dict[idx]['scenario']] 
                            + scenario_lines[x0s_dict[idx]['scenario']] 
                        ) - len(scenario_baseload[x0s_dict[idx]['scenario']]) - len(scenario_minor_lines[x0s_dict[idx]['scenario']])
        
        if ((len(x0) != bound_length) and not (isinstance(x0, float) and np.isnan(x0))):
            print(f"Initial guess 'x_0' for scenario {x0s_dict[idx]['scenario']} contains {len(x0)} elements, but the differential evolution bounds contain {bound_length} elements.")
            flag = False

    for scenario in scenarios_list:
        if scenario not in initial_guess_scenarios:
            print(f"'scenario' {scenario} is defined in scenarios.csv, but does not have an entry in initial_guess.csv")
            flag = False

    return flag

def validate_config(model_data : ModelData) -> bool:
    config_flag = True
    if not validate_model_config(model_data.config):
        print('config.csv contains errors.')    
        config_flag =  False
    else:
        print('config.csv validated!')

    scenarios_list, scenario_nodes, scenario_lines, flag = validate_scenarios(model_data.scenarios)
    if not flag:
        print('scenarios.csv contains errors.')
        config_flag =  False
    else:
        print('scenarios.csv validated!')

    scenario_fuels, flag = validate_fuels(model_data.fuels, scenarios_list)
    if not flag:
        print('fuels.csv contains errors.')
        config_flag =  False
    else:
        print('fuels.csv validated!')

    scenario_lines, scenario_minor_lines, flag = validate_lines(model_data.lines, scenarios_list, scenario_nodes)
    if not flag:
        print('lines.csv contains errors.')
        config_flag =  False
    else:
        print('lines.csv validated!')

    scenario_generators, scenario_baseload, flag = validate_generators(model_data.generators, scenarios_list, scenario_fuels, scenario_lines, scenario_nodes)
    if not flag:
        print('generators.csv contains errors.')
        config_flag =  False
    else:
        print('generators.csv validated!')

    scenario_storages, flag = validate_storages(model_data.storages, scenarios_list, scenario_nodes, scenario_lines)
    if not flag:
        print('storages.csv contains errors.')
        config_flag =  False
    else:
        print('storages.csv validated!')

    if not validate_initial_guess(model_data.x0s, scenarios_list, scenario_generators, scenario_storages, scenario_lines, scenario_baseload, scenario_minor_lines):
        print('initial_guess.csv contains errors.')
        config_flag =  False
    else:
        print('initial_guess.csv validated!')

    return config_flag

def validate_data():
    """ if not validate_datafiles_config():
        print('datafiles.csv contains errors.')

    if not validate_electricity():
        print('Demand profiles contain errors.')
    
    if not validate_generation():
        print('cGeneration traces contain errors.')

    if not validate_flexible_limits():
        print('Flexible limits contains errors.') """

    return True