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
    from numba.core.types import DictType, int64, UniTuple, ListType, float64, unicode_type, boolean
else:
    class _Int64:
        @classmethod
        def __class_getitem__(cls, key):
            return NDArray[np.int64]

    class _Float64:
        @classmethod
        def __class_getitem__(cls, key):
            return NDArray[np.float64]

    class _Boolean:
        @classmethod
        def __class_getitem__(cls, key):
            return NDArray[np.bool_]

    class _Unicode:
        @classmethod
        def __class_getitem__(cls, key):
            return NDArray[np.unicode_]

    int64 = _Int64
    float64 = _Float64
    boolean = _Boolean
    unicode_type = _Unicode

    def UniTuple(ty, n: int):
        _map = {
            _Float64: float,
            _Int64: int,
            _Boolean: bool,
            _Unicode: str,
            float: float,
            int: int,
            bool: bool,
            str: str
        }
        base = _map.get(ty, ty)
        return Tuple[tuple([base]*n)]

    def DictType(key_ty, val_ty):
        try:
            return Dict[key_ty, val_ty]
        except Exception:
            return Dict

    def ListType(val_ty):
        try:
            return List[val_ty]
        except Exception:
            return List

    class TypedDict(dict):
        def __init__(self, key_type=None, value_type=None):
            super().__init__()
            self.key_type = key_type
            self.value_type = value_type

        @staticmethod
        def empty(key_type=None, value_type=None):
            return TypedDict(key_type, value_type)

    class TypedList(list):
        def __init__(self, value_type=None):
            super().__init__()
            self.value_type = value_type

        @staticmethod
        def empty_list(value_type=None):
            return TypedList(value_type)
