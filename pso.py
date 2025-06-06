import numpy as np
import math
import matplotlib.pyplot as plt

class Particle:
    def __init__(self, dim, bounds):
        self.position = np.random.uniform(bounds[0], bounds[1], dim)
        self.velocity = np.zeros(dim)
        self.best_position = self.position.copy()
        self.best_score = float('inf')

class PSO:
    def __init__(self, objective_func, bounds, dim, n_particles=100, max_iter=1000, w=0.7, c1=1.5, c2=1.5):
        self.objective_func = objective_func
        self.bounds = bounds
        self.dim = dim
        self.n_particles = n_particles
        self.max_iter = max_iter
        self.w = w  # inercia
        self.c1 = c1  # propio
        self.c2 = c2  # social
        self.particles = [Particle(self.dim, bounds) for _ in range(n_particles)]
        self.global_best_position = None
        self.global_best_score = float('inf')

    def optimize(self):
        for _ in range(self.max_iter):
            for particle in self.particles:
                score = self.objective_func(particle.position)
                
                if score < particle.best_score:
                    particle.best_position = particle.position.copy()
                    particle.best_score = score
                    
                    if score < self.global_best_score:
                        self.global_best_position = particle.position.copy()
                        self.global_best_score = score

                r1, r2 = np.random.rand(2)
                particle.velocity = (self.w * particle.velocity + 
                                  self.c1 * r1 * (particle.best_position - particle.position) +
                                  self.c2 * r2 * (self.global_best_position - particle.position))
                
                particle.position += particle.velocity
                
                particle.position = np.clip(particle.position, self.bounds[0], self.bounds[1])

        return self.global_best_position, self.global_best_score

def f1(x):
    return 4 - 4 * x[0]**3 - 4 * x[0] + x[1]**2

def f2(x):
    return (1/899) * (sum(x[i]**2 * 2**(i+1) for i in range(6)) - 1745)

def f3(x):
    return (x[0]**6 + x[1]**4 + 12)**2 + (2*x[0] + x[1] - 4)**2

def f4(x):
    if any(xi <= 2 or xi >= 10 for xi in x):
        return float('inf')
    
    try:
        sum_term = sum((math.log(x[i]-2))**2 + (math.log(10-x[i]))**2 for i in range(10))
        prod_term = (np.prod(x))**0.2
        return sum_term - prod_term
    except (ValueError, ZeroDivisionError):
        return float('inf')

# Function to run multiple iterations
def run_multiple_iterations(func, bounds, dim, config, n_iterations=10):
    results = []
    for i in range(n_iterations):
        pso = PSO(func, bounds, dim, n_particles=30, max_iter=100, 
                 w=config['w'], c1=config['c1'], c2=config['c2'])
        best_pos, best_score = pso.optimize()
        results.append((best_pos, best_score))
        print(f"Config {config['name']} - Iter {i+1}: Mejor pos = {best_pos}, Mejor valor = {best_score}")
    return results

def plot_results(results_dict, function_name):
    plt.figure(figsize=(12, 6))
    
    for config_name, results in results_dict.items():
        scores = [score for _, score in results]
        plt.plot(range(1, 11), scores, 'o-', label=config_name)
    
    plt.title(f'Resultados del PSO con la función {function_name} con diferentes configuraciones')
    plt.xlabel('Iteration')
    plt.ylabel('Best Score')
    plt.grid(True)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    # Conf
    configurations = [
        {'name': 'Estandar', 'w': 0.7, 'c1': 1.5, 'c2': 1.5},
        {'name': 'Alta Exploracion', 'w': 0.9, 'c1': 2.0, 'c2': 1.0},
        {'name': 'Alta Explotation', 'w': 0.4, 'c1': 1.0, 'c2': 2.0},
        {'name': 'Balanceado', 'w': 0.6, 'c1': 1.7, 'c2': 1.7},
        {'name': 'Conservativo', 'w': 0.5, 'c1': 1.0, 'c2': 1.0}
    ]

    bounds_f1 = (-5, 5)
    bounds_f2 = (0, 1)
    bounds_f3 = (-500, 500)
    bounds_f4 = (2.001, 10)

    functions = [
        ('f1', f1, bounds_f1, 2),
        ('f2', f2, bounds_f2, 6),
        ('f3', f3, bounds_f3, 2),
        ('f4', f4, bounds_f4, 10)
    ]

    for func_name, func, bounds, dim in functions:
        print(f"\nProbando {func_name} con las 5 configuraciones:")
        results_dict = {}
        
        for config in configurations:
            print(f"\nConfigución: {config['name']}")
            results = run_multiple_iterations(func, bounds, dim, config)
            results_dict[config['name']] = results
        
        plot_results(results_dict, func_name) 