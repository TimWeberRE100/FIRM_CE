import numpy as np

from firm_ce.system.scenario import Scenario

def construct_scenario_class(scenario):
    MLoad = np.array(
            [scenario.nodes[idx].demand_data
            for idx in range(0, max(scenario.nodes)+1)],
            dtype=np.float64, ndmin=2
        ).T

    years = scenario.final_year - scenario.first_year + 1                                

    leap_years = 0
    for y in range(scenario.first_year, scenario.first_year + years):
        if y % 4 == 0 and (y % 100 != 0 or y % 400 == 0):
            leap_years += 1
        
    return Scenario(scenario.intervals,
                        years,
                        len(scenario.nodes),
                        len(scenario.lines),
                        scenario.resolution,
                        MLoad.sum() / years,
                        0.0,
                        scenario.allowance,
                        (years+leap_years/365)/years 
                        )