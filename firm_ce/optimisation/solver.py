from typing import Dict, List

class Solution:
    def __init__(self, x):
        self.x = x
        
        self.intervals = intervals
        self.nodes = nodes
        self.years = years
        self.resolution = resolution
        self.network, self.directconns = network, directconns
        self.trans_tdc_mask = trans_tdc_mask
        self.hvdc_mask = hvdc_mask
        self.Windl_Viet_int = Windl_Viet_int
        
        self.MLoad = MLoad

        self.CPV = x[: pidx]  # CPV(i), GW
        self.CWind = x[pidx: widx]  # CWind(i), GW

        _CInter = x[bhidx:iidx]
        CInter = np.zeros(len(CInter_mask), dtype=np.float64)
        counter = 0
        for i in range(len(CInter)):
            if CInter_mask[i] == 1:
                CInter[i] = _CInter[counter]
                counter+=1
        self.CInter = CInter
        
        # Manually replicating np.tile functionality for CPV and CWind
        CPV_tiled = np.zeros((intervals, len(self.CPV)), dtype=np.float64)
        CWind_tiled = np.zeros((intervals, len(self.CWind)), dtype=np.float64)
        for i in range(intervals):
            for j in range(len(self.CPV)):
                CPV_tiled[i, j] = self.CPV[j]
            for j in range(len(self.CWind)):
                CWind_tiled[i, j] = self.CWind[j]

        GPV = TSPV * CPV_tiled * 1000.  # GPV(i, t), GW to MW
        GWind = TSWind * CWind_tiled * 1000.  # GWind(i, t), GW to MW
        self.GWind_sites = GWind

        self.GPV, self.GWind = np.empty((intervals, nodes), np.float64), np.empty((intervals, nodes), np.float64)
        for i, j in enumerate(Nodel_int):
            self.GPV[:,i] = GPV[:, PVl_int==j].sum(axis=1)
            self.GWind[:,i] = GWind[:, Windl_int==j].sum(axis=1) 
        
        self.CPHP = x[widx: spidx]  # CPHP(j), GW
        self.CPHS = x[spidx: seidx]  # S-CPHS(j), GWh
        self.CBP = x[seidx: bpidx] # GW
        self.CBH = x[bpidx: bhidx] # hours
        self.CBS = self.CBP * self.CBH # GWh

        self.DCloss = DCloss
        self.CHVDC = x[iidx:] # GW
        self.DCdistance = DCdistance
        
        self.efficiency = efficiency

        self.baseload = baseload # MWh
        self.hydro_baseload = hydro_baseload
        self.CPeak = CPeak # GW
        self.CHydro = CHydro # GW
        self.CBaseload = CBaseload # GW
        
        self.UnitCosts = UnitCosts
        
        self.allowance = allowance
        
        self.evaluated=False
        
    def _evaluate(self):
        self.Lcoe, self.Penalties = F(self)
        self.evaluated=True

class Solver:
    def __init__(self, type: str, config: Dict[str, str], x: List[float]) -> None:
        self.type = type
        self.mutations = config['mutations']
        self.population = config['population']
        self.recombination = config['recombination']
        self.iterations = config['iterations']
        self.solution = Solution(x)

    def single_time():
        pass

    def capacity_expansion():
        pass

    def solve():
        pass