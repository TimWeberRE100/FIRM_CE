import numpy as np
import os

from firm_ce.io.file_manager import import_config_csvs
from firm_ce.common.helpers import parse_comma_separated
from firm_ce.common.logging import init_model_logger


class ModelData:
    def __init__(self) -> None:
        objects = import_config_csvs()

        try:
            config_dict = objects.get('config')
            for v in config_dict.values():
                if v.get('name') == 'model_name':
                    model_name = v.get('value')
                    break
        except:
            model_name = 'Model'

        self.logger, self.results_dir = init_model_logger(model_name)

        self.scenarios = objects.get('scenarios')
        self.generators = objects.get('generators')
        self.fuels = objects.get('fuels')
        self.lines = objects.get('lines')
        self.storages = objects.get('storages')
        self.config = objects.get('config')
        self.x0s = objects.get('initial_guess')
        self.datafiles = objects.get('datafiles')

    def validate(self):
        return validate_config(self)


def validate_range(val, min_val, max_val=None, inclusive=True):
    try:
        val = float(val)
        if inclusive:
            return min_val <= val <= max_val if max_val is not None else min_val <= val
        else:
            return min_val < val < max_val if max_val is not None else min_val < val
    except (TypeError, ValueError):
        return False


def validate_positive_int(val):
    try:
        return int(val) > 0
    except (TypeError, ValueError):
        return False


def validate_enum(val, options):
    return val in options


def parse_list(val):
    return parse_comma_separated(val) if not is_nan(val) else []


def is_nan(val):
    return isinstance(val, float) and np.isnan(val)


def validate_model_config(config_dict, model_logger):
    flag = True
    validators = {
        'mutation': lambda v: validate_range(v, 0, 2, inclusive=False),
        'iterations': validate_positive_int,
        'population': validate_positive_int,
        'recombination': lambda v: validate_range(v, 0, 1),
        'type': lambda v: validate_enum(v, ['single_time', 'capacity_expansion', 'near_optimum', 'midpoint_explore'],),
        'model_name': None,
        'near_optimal_tol': lambda v: validate_range(v, 0, 1),
        'midpoint_count': validate_positive_int,
        'balancing_type': lambda v: validate_enum(v, ['simple', 'full'],),
        'simple_blocks_per_day': validate_positive_int,
        'fixed_costs_threshold': lambda v: validate_range(v, 0),
    }

    for item in config_dict.values():
        name = item.get('name')
        value = item.get('value')

        if name not in validators:
            model_logger.warning(f"Unknown configuration name {name}")
            continue

        if not validators[name]:
            continue

        try:
            if not validators[name](value):
                model_logger.error("Invalid value for '%s': %s", name, value)
                flag = False
        except Exception as e:
            model_logger.exception("Exception during validation of '%s': %s", name, e)
            flag = False

    return flag


def validate_scenarios(scenarios_dict, model_logger):
    flag = True
    scenarios_list = []
    scenario_nodes = {}
    scenario_lines = {}
    firstyear = finalyear = None

    for item in scenarios_dict.values():
        name = item['scenario_name']
        if name in scenarios_list:
            model_logger.error("Duplicate scenario name '%s'", name)
            flag = False
        scenarios_list.append(name)

        if not validate_range(item['resolution'], 0):
            model_logger.error("'resolution' must be float greater than 0")
            flag = False

        if not validate_range(item['allowance'], 0, 1):
            model_logger.error("'allowance' must be float in range [0,1]")
            flag = False

        try:
            fy = int(item['firstyear'])
            ly = int(item['finalyear'])
            firstyear = fy if firstyear is None else min(firstyear, fy)
            finalyear = ly if finalyear is None else max(finalyear, ly)
        except ValueError:
            model_logger.error("'firstyear' and 'finalyear' must be integers")
            flag = False

        scenario_nodes[name] = parse_list(item.get('nodes'))
        scenario_lines[name] = parse_list(item.get('lines'))

    if firstyear is not None and finalyear is not None and firstyear > finalyear:
        model_logger.error("'firstyear' must be less than or equal to 'finalyear'")
        flag = False

    return scenarios_list, scenario_nodes, scenario_lines, flag


def validate_fuels(fuels_dict, scenarios_list, model_logger):
    flag = True
    scenario_fuels = {scenario: [] for scenario in scenarios_list}

    for idx, item in fuels_dict.items():
        if not validate_range(item['emissions'], 0):
            model_logger.error("'emissions' must be float greater than or equal to 0")
            flag = False

        if not validate_range(item['cost'], 0):
            model_logger.error("'cost' must be float greater than or equal to 0")
            flag = False

        for scenario in parse_list(item.get('scenarios')):
            if scenario in scenarios_list:
                scenario_fuels[scenario].append(item['name'])
            else:
                model_logger.warning("'scenario' %s for fuel.id %s not defined in scenarios.csv", scenario, idx)

    return scenario_fuels, flag


def validate_lines(lines_dict, scenarios_list, scenario_nodes, model_logger):
    flag = True
    scenario_lines = {s: [] for s in scenarios_list}
    scenario_minor_lines = {s: [] for s in scenarios_list}

    for idx, item in lines_dict.items():
        numeric_fields = {
            'length': int,
            'capex': float,
            'transformer_capex': float,
            'fom': float,
            'vom': float,
            'lifetime': int,
            'discount_rate': float,
            'loss_factor': float,
            'initial_capacity': float,
            'max_build': float,
            'min_build': float,
        }

        for field, cast in numeric_fields.items():
            try:
                val = cast(item[field])
                if 'discount_rate' == field:
                    if not (0 <= val <= 1):
                        raise ValueError
                elif 'loss_factor' == field:
                    if not (0 <= val < 1):
                        raise ValueError
                else:
                    if val < 0:
                        raise ValueError
            except ValueError:
                model_logger.error("'%s' must be a valid %s in appropriate range", field, cast.__name__)
                flag = False

        if float(item['min_build']) > float(item['max_build']):
            model_logger.error("'min_build' must be less than 'max_build'")
            flag = False

        for scenario in parse_list(item.get('scenarios')):
            if scenario in scenarios_list:
                scenario_lines[scenario].append(item['name'])

                for endpoint in ['node_start', 'node_end']:
                    node_val = item.get(endpoint)
                    if (node_val not in scenario_nodes[scenario]) and not is_nan(node_val):
                        model_logger.error("'%s' %s for line %s is not defined in scenario %s", endpoint, node_val, item['name'], scenario)
                        flag = False

                if any(is_nan(item.get(n)) for n in ['node_start', 'node_end']):
                    scenario_minor_lines[scenario].append(item['name'])
            else:
                model_logger.warning("'scenario' %s for line.id %s not defined in scenarios.csv", scenario, idx)

    return scenario_lines, scenario_minor_lines, flag


def validate_generators(generators_dict, scenarios_list, scenario_fuels, scenario_lines, scenario_nodes, model_logger):
    flag = True
    scenario_generators = {s: [] for s in scenarios_list}
    scenario_baseload = {s: [] for s in scenarios_list}

    fields = ['capex', 'fom', 'vom', 'heat_rate_base', 'heat_rate_incr', 'initial_capacity', 'max_build', 'min_build']

    for idx, item in generators_dict.items():
        for field in fields:
            if not validate_range(item[field], 0):
                model_logger.error("'%s' must be float greater than or equal to 0", field)
                flag = False

        if not validate_range(item['discount_rate'], 0, 1):
            model_logger.error("'discount_rate' must be float in range [0,1]")
            flag = False

        if not validate_enum(item['unit_type'], ['solar', 'wind', 'flexible', 'baseload']):
            model_logger.error("'unit_type' must be one of ['solar', 'wind', 'flexible', 'baseload']")
            flag = False

        if float(item['min_build']) > float(item['max_build']):
            model_logger.error("'min_build' must be less than 'max_build'")
            flag = False

        for scenario in parse_list(item.get('scenarios')):
            if scenario in scenarios_list:
                if item['name'] in scenario_generators[scenario]:
                    model_logger.error("Duplicate generator name '%s' in scenario %s", item['name'], scenario)
                    flag = False
                else:
                    scenario_generators[scenario].append(item['name'])

                if item['unit_type'] == 'baseload':
                    scenario_baseload[scenario].append(item['name'])

                if item['node'] not in scenario_nodes[scenario]:
                    model_logger.error("'node' %s for generator %s is not defined in scenario %s", item['node'], item['name'], scenario)
                    flag = False

                if item['fuel'] not in scenario_fuels[scenario]:
                    model_logger.error("'fuel' %s for generator %s is not defined in scenario %s", item['fuel'], item['name'], scenario)
                    flag = False

                if item['line'] not in scenario_lines[scenario]:
                    model_logger.error("'line' %s for generator %s is not defined in scenario %s", item['line'], item['name'], scenario)
                    flag = False
            else:
                model_logger.warning("'scenario' %s for generator.id %s not defined in scenarios.csv", scenario, idx)

    return scenario_generators, scenario_baseload, flag


def validate_storages(storages_dict, scenarios_list, scenario_nodes, scenario_lines, model_logger):
    flag = True
    scenario_storages = {s: [] for s in scenarios_list}

    fields = [
        'capex_p',
        'capex_e',
        'fom',
        'vom',
        'initial_power_capacity',
        'initial_energy_capacity',
        'max_build_p',
        'min_build_p',
        'max_build_e',
        'min_build_e',
    ]

    for idx, item in storages_dict.items():
        for field in fields:
            if not validate_range(item[field], 0):
                model_logger.error("'%s' must be float >= 0", field)
                flag = False

        for bounded in [('min_build_p', 'max_build_p'), ('min_build_e', 'max_build_e')]:
            if float(item[bounded[0]]) > float(item[bounded[1]]):
                model_logger.error("'%s' must be <= '%s'", bounded[0], bounded[1])
                flag = False

        for field in ['lifetime', 'duration']:
            try:
                if int(item[field]) < 0:
                    raise ValueError
            except:
                model_logger.error("'%s' must be int >= 0", field)
                flag = False

        for efficiency in ['charge_efficiency', 'discharge_efficiency']:
            if not validate_range(item[efficiency], 0, 1):
                model_logger.error("'%s' must be float in [0,1]", efficiency)
                flag = False

        if not validate_range(item['discount_rate'], 0, 1):
            model_logger.error("'discount_rate' must be float in [0,1]")
            flag = False

        for scenario in parse_list(item.get('scenarios')):
            if scenario in scenarios_list:
                if item['name'] in scenario_storages[scenario]:
                    model_logger.error("Duplicate storage name '%s' in scenario %s", item['name'], scenario)
                    flag = False
                else:
                    scenario_storages[scenario].append(item['name'])

                if item['node'] not in scenario_nodes[scenario]:
                    model_logger.error(
                        "'node' %s for storage %s is not defined in scenario %s",
                        item['node'],
                        item['name'],
                        scenario,
                    )
                    flag = False

                if item['line'] not in scenario_lines[scenario]:
                    model_logger.error(
                        "'line' %s for storage %s is not defined in scenario %s",
                        item['line'],
                        item['name'],
                        scenario
                    )
                    flag = False
            else:
                model_logger.warning("'scenario' %s for storage.id %s not defined in scenarios.csv", scenario, idx)

    return scenario_storages, flag


def validate_initial_guess(
    x0s_dict,
    scenarios_list,
    scenario_generators,
    scenario_storages,
    scenario_lines,
    scenario_baseload,
    scenario_minor_lines,
    model_logger
):
    flag = True
    initial_guess_scenarios = []

    for idx, item in x0s_dict.items():
        scenario = item['scenario']

        if scenario not in scenarios_list:
            model_logger.warning("'scenario' %s in initial_guess.csv not defined in scenarios.csv", scenario)

        initial_guess_scenarios.append(scenario)

        x0 = parse_list(item['x_0'])
        bound_length = len(
            scenario_generators[scenario]
            + scenario_storages[scenario]
            + scenario_storages[scenario]
            + scenario_lines[scenario]
        ) - len(scenario_minor_lines[scenario])

        if x0 and not (len(x0) == bound_length):
            model_logger.error(
                "Initial guess 'x_0' for scenario %s contains %d elements, expected %d",
                scenario,
                len(x0),
                bound_length,
            )
            flag = False

    for scenario in scenarios_list:
        if scenario not in initial_guess_scenarios:
            model_logger.error("'scenario' %s is defined in scenarios.csv but missing from initial_guess.csv", scenario)
            flag = False

    return flag


def validate_datafiles_config(scenario_filenames, scenario_datafile_types, model_logger):
    valid_types = {'demand', 'generation', 'flexible_annual_limit'}
    all_filenames = set(os.listdir("firm_ce/data"))
    flag = True

    for fn in scenario_filenames:
        if fn not in all_filenames:
            model_logger.error(f"Missing file data/{fn}")
            flag = False

    for dtype in scenario_datafile_types:
        if not dtype or dtype not in valid_types:
            model_logger.error(f"Invalid or missing datafile_type '{dtype}'")
            flag = False

    return flag


def validate_electricity(node_list, model_logger):
    flag = True
    return flag


def validate_generation(solar_list, wind_list, baseload_list, model_logger):
    flag = True
    return flag


def validate_flexible_limits(flexible_list, model_logger):
    flag = True
    return flag


def validate_config(model_data: ModelData) -> bool:
    config_flag = True
    model_logger = model_data.logger

    if not validate_model_config(model_data.config, model_logger):
        model_logger.error("config.csv contains errors.")
        config_flag = False
    else:
        model_logger.info("config.csv validated!")

    scenarios_list, scenario_nodes, scenario_lines, flag = validate_scenarios(model_data.scenarios, model_logger)
    if not flag:
        model_logger.error("scenarios.csv contains errors.")
        config_flag = False
    else:
        model_logger.info("scenarios.csv validated!")

    scenario_fuels, flag = validate_fuels(model_data.fuels, scenarios_list, model_logger)
    if not flag:
        model_logger.error("fuels.csv contains errors.")
        config_flag = False
    else:
        model_logger.info("fuels.csv validated!")

    scenario_lines, scenario_minor_lines, flag = validate_lines(
        model_data.lines,
        scenarios_list,
        scenario_nodes,
        model_logger
    )
    if not flag:
        model_logger.error("lines.csv contains errors.")
        config_flag = False
    else:
        model_logger.info("lines.csv validated!")

    scenario_generators, scenario_baseload, flag = validate_generators(
        model_data.generators,
        scenarios_list,
        scenario_fuels,
        scenario_lines,
        scenario_nodes,
        model_logger,
    )
    if not flag:
        model_logger.error("generators.csv contains errors.")
        config_flag = False
    else:
        model_logger.info("generators.csv validated!")

    scenario_storages, flag = validate_storages(
        model_data.storages,
        scenarios_list,
        scenario_nodes,
        scenario_lines,
        model_logger,
    )
    if not flag:
        model_logger.error("storages.csv contains errors.")
        config_flag = False
    else:
        model_logger.info("storages.csv validated!")

    if not validate_initial_guess(
        model_data.x0s,
        scenarios_list,
        scenario_generators,
        scenario_storages,
        scenario_lines,
        scenario_baseload,
        scenario_minor_lines,
        model_logger,
    ):
        model_logger.error("initial_guess.csv contains errors.")
        config_flag = False
    else:
        model_logger.info("initial_guess.csv validated!")

    return config_flag


def validate_data(all_datafiles, scenario_name, model_logger):
    flag = True
    scenario_filenames = []
    scenario_datafile_types = []

    for item in all_datafiles.values():
        scenario_list = parse_comma_separated(item['scenarios'])
        if scenario_name in scenario_list:
            scenario_filenames.append(item['filename'])
            scenario_datafile_types.append(item['datafile_type'])

    if not validate_datafiles_config(scenario_filenames, scenario_datafile_types, model_logger):
        model_logger.error(f'datafiles.csv contains errors for scenario {scenario_name}.')
        flag = False
    else:
        model_logger.info(f'datafiles.csv validated for scenario {scenario_name}!')

    """ if not validate_electricity(model_logger):
        model_logger.error(f'Demand profiles contain errors for scenario {scenario_name}.')
        flag = False
    else:
        model_logger.info(f'Demand profiles validated for scenario {scenario_name}!')

    if not validate_generation(model_logger):
        model_logger.error(f'Generation traces contain errors for scenario {scenario_name}.')
        flag = False
    else:
        model_logger.info(f'Generation traces validated for scenario {scenario_name}!')

    if not validate_flexible_limits(model_logger):
        model_logger.error(f'Flexible limits contains errors for scenario {scenario_name}.')
        flag = False
    else:
        model_logger.info(f'Flexible limits validated for scenario {scenario_name}!') """

    return flag
