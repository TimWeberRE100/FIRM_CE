import time

from src.firm_ce.model import Model

start_time = time.time()
model = Model()
model_build_time = time.time()

print(model.scenarios)
print(f"Model build time: {model_build_time - start_time:.4f} seconds")

model.solve()
end_time = time.time()
print(f"Model solve time: {end_time - model_build_time:.4f} seconds")
