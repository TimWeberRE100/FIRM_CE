asset_types = ('generator','storage','load','spillage','deficit')
generator_unit_types = ('solar','wind','flexible','baseload')
storage_unit_types = ('bess','phes')
line_unit_types = ('hvdc','hvac','submarine')
trace_types = ('power','remaining_energy')

SETTINGS = {
    'asset_types': tuple((i, v) for i, v in enumerate(asset_types)),
    'generator_unit_types': tuple((i, v) for i, v in enumerate(generator_unit_types)),
    'storage_unit_types': tuple((i, v) for i, v in enumerate(storage_unit_types)),
    'line_unit_types': tuple((i, v) for i, v in enumerate(line_unit_types)),
    'trace_types': tuple((i, v) for i, v in enumerate(trace_types)),
}