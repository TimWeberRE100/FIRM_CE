"""
An example is provided for building a FIRM Model instance and then solves a simulates a single solution

Alternative filepaths for the config and data folders can be provided as arguments to the Model instantiation.
"""

import time

from firm_ce.model import Model

start_time = time.time()
model = Model()
model_build_time = time.time()

print(model.scenarios)
print(f"Model build time: {model_build_time - start_time:.4f} seconds")

model.solve()
end_time = time.time()
print(f"Model solve time: {end_time - model_build_time:.4f} seconds")
