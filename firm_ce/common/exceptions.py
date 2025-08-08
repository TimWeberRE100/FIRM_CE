from firm_ce.common.constants import JIT_ENABLED
if JIT_ENABLED:
    from numba import njit
else:
    def njit(func=None, **kwargs):
        if func is not None:
            return func
        def wrapper(f):
            return f
        return wrapper

@njit
def raise_static_modification_error():
    raise ValueError("Attempting to modify a static jitclass instance. Use the create_dynamic_copy method within the worker process to modify attributes.")

@njit
def raise_getting_unloaded_data_error():
    raise RuntimeError("Attempting to get data with status 'unloaded'.")

@njit
def raise_unknown_balancing_type_error():
    raise RuntimeError("Unknown value for balancing_type in config.csv. Valid values are 'simple' and 'full'.")