import numpy as np

# funciones objetivo
def f1(x):
    return 4 - 4*x[0]**3 - 4*x[0] + x[1]**2

def f2(x):
    powers = np.array([2**i for i in range(1, 7)])
    return (np.sum((x[:6]**2) * powers) - 1745) / 899

def f3(x):
    return (x[0]**6 + x[1]**4 + 12)**2 + (2*x[0] + x[1] - 4)**2

def f4(x):
    if np.any(x <= 2) or np.any(x >= 10):
        return np.inf  # Penalize invalid values
    log_terms = np.log(x - 2)**2 + np.log(10 - x)**2
    prod_term = np.prod(x)**0.2
    return np.sum(log_terms) - prod_term

# funciones y variables
FUNCIONES = {
    "f1": {
        "func": f1,
        "name": "Función 1",
        "dim": 2,
        "bounds": (-5, 5)
    },
    "f2": {
        "func": f2,
        "name": "Función 2",
        "dim": 6,
        "bounds": (0, 1)
    },
    "f3": {
        "func": f3,
        "name": "Función 3",
        "dim": 2,
        "bounds": (-500, 500)
    },
    "f4": {
        "func": f4,
        "name": "Función 4",
        "dim": 10,
        "bounds": (-2001, 10)  # ¡Ojo f4 requiere x_i > 2
    }
}

def evaluar_funcion(f_config, x):
    func = f_config["func"]
    return func(np.array(x))

def inicializar_particula(bounds, dim):
    low, high = bounds
    return np.random.uniform(low, high, size=dim)
