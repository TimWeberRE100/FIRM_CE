import numpy as np
from numpy.typing import NDArray
from typing import Tuple, Dict, List

from firm_ce.common.constants import JIT_ENABLED

EvaluationRecord_Type = Tuple[str, str, float, float, float, NDArray[np.float64]]
BroadOptimumVars_Type = Tuple[int, str, bool, str]
BandCandidates_Type = Dict[str, Tuple[List[float], List[float]]]

if JIT_ENABLED:
    from numba.typed.typeddict import Dict as TypedDict
    from numba.typed.typedlist import List as TypedList
    from numba.core.types import DictType, int64, UniTuple, ListType, float64, string, boolean
else:
    def DictType(*args):
        return Dict 
    def ListType(*args):
        return List  
    def UniTuple(*args):
        return Tuple         
    int64 = int
    float64 = float
    string = str
    boolean = bool

    class TypedDict:
        def __init__(self, key_type, value_type):
            self.key_type
            self.value_type
        @staticmethod
        def empty(key_type, value_type):
            return {}
        
    class TypedList:
        def __init__(self, value_type):
            self.key_type
            self.value_type
        @staticmethod
        def empty_list(value_type):
            return []