from firm_ce import Model
import time

start_time = time.time()
model = Model()
model_build_time = time.time()

print(model.scenarios)
print(f"Model build time: {model_build_time - start_time:.4f} seconds")

#print("\n=== default model: near_optimum ===")
model.solve()
end_time = time.time()
print(f"Model solve time: {end_time - model_build_time:.4f} seconds")

""" 
model.config.type = "midpoint_explore"
print("\n=== running midpoint_explore ===")
t1 = time.time()
model.solve()
print(f"midpoint_explore done in {time.time()-t1:.1f}s") """