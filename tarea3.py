import numpy as np
import pandas as pd

# Función 1: f1(x) = 4 - 4x1^3 - 4x1 + x2^2
def f1_real(x):
    x1, x2 = x[0], x[1]
    return 4 - 4 * x1**3 - 4 * x1 + x2**2

# Función 2: f2(x) = (1/899) * (sum(xi^2 * 2^i) - 1745)
def f2_real(x):
    total = sum(x[i]**2 * 2**(i) for i in range(6))
    return (1 / 899) * (total - 1745)

# Función 3: f3(x) = (x1^6 + x2^4 - 17)^2 + (2x1 + x2 - 4)^2
def f3_real(x):
    x1, x2 = x[0], x[1]
    return (x1**6 + x2**4 - 17)**2 + (2*x1 + x2 - 4)**2

# Función 4: f4(x) = sum[(ln(xi - 2))^2 + (ln(10 - xi))^2] - (prod xi)^0.2
def f4_real(x):
    if np.any(x <= 2) or np.any(x >= 10):
        return float('inf')  # Penaliza fuera de dominio
    sum_logs = sum((np.log(xi - 2))**2 + (np.log(10 - xi))**2 for xi in x)
    prod = np.prod(x)
    return sum_logs - prod**0.2

class Particle:
    def __init__(self, bounds):
        self.position = np.array([np.random.uniform(low, high) for low, high in bounds])
        self.velocity = np.zeros_like(self.position)
        self.best_position = self.position.copy()
        self.best_score = float('inf')

    def update_velocity(self, global_best, w, c1, c2):
        r1 = np.random.rand(len(self.position))
        r2 = np.random.rand(len(self.position))
        cognitive = c1 * r1 * (self.best_position - self.position)
        social = c2 * r2 * (global_best - self.position)
        self.velocity = w * self.velocity + cognitive + social

    def update_position(self, bounds):
        self.position += self.velocity
        for i, (low, high) in enumerate(bounds):
            self.position[i] = np.clip(self.position[i], low, high)

class PSO:
    def __init__(self, objective_function, bounds, num_particles=15, max_iter=50, w_max=0.9, w_min=0.3, c1=1.5, c2=1.5):
        self.objective_function = objective_function
        self.bounds = bounds
        self.num_particles = num_particles
        self.max_iter = max_iter
        self.w_max = w_max
        self.w_min = w_min
        self.c1 = c1
        self.c2 = c2
        self.swarm = [Particle(bounds) for _ in range(num_particles)]
        self.global_best = None
        self.global_best_score = float('inf')

    def optimize(self):
        for t in range(self.max_iter):
            w = self.w_max - t * (self.w_max - self.w_min) / self.max_iter
            for particle in self.swarm:
                score = self.objective_function(particle.position)
                if score < particle.best_score:
                    particle.best_score = score
                    particle.best_position = particle.position.copy()
                if score < self.global_best_score:
                    self.global_best_score = score
                    self.global_best = particle.position.copy()
            for particle in self.swarm:
                particle.update_velocity(self.global_best, w, self.c1, self.c2)
                particle.update_position(self.bounds)
        return self.global_best, self.global_best_score

funciones = [
    ("f1", f1_real, [(-5, 5), (-5, 5)]),
    ("f2", f2_real, [(0, 1)] * 6),
    ("f3", f3_real, [(-500, 500), (-500, 500)]),
    ("f4", f4_real, [(2.001, 10)] * 10)
]

configs = [
    {"w_max": 0.9, "w_min": 0.3, "c1": 1.5, "c2": 1.5},
    {"w_max": 0.7, "w_min": 0.4, "c1": 1.7, "c2": 1.7},
    {"w_max": 1.0, "w_min": 0.5, "c1": 1.4, "c2": 1.4},
    {"w_max": 0.8, "w_min": 0.2, "c1": 2.0, "c2": 2.0},
]

for nombre_func, func, bounds in funciones:
    print(f"\n========== Ejecutando PSO para {nombre_func} ==========")
    results = []
    for idx, cfg in enumerate(configs, 1):
        for run in range(1, 11):
            pso = PSO(objective_function=func, bounds=bounds, num_particles=15, max_iter=50,
                      w_max=cfg["w_max"], w_min=cfg["w_min"], c1=cfg["c1"], c2=cfg["c2"])
            best_position, best_score = pso.optimize()
            results.append({
                "Configuración": f"C{idx} (w={cfg['w_max']}/{cfg['w_min']}, c1={cfg['c1']}, c2={cfg['c2']})",
                "Ejecución": run,
                "Mejor Score": best_score
            })

    df = pd.DataFrame(results)
    for cfg in df["Configuración"].unique():
        sub_df = df[df["Configuración"] == cfg]
        print(f"\n--- Resultados para {cfg} ---")
        print(sub_df[["Ejecución", "Mejor Score"]].to_string(index=False))

    print("\n=== Resumen estadístico por configuración ===")
    summary = df.groupby("Configuración")["Mejor Score"].agg(["min", "max", "mean"]).round(4)
    print(summary.reset_index().to_string(index=False))
