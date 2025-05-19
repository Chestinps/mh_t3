import numpy as np
from funciones import evaluar_funcion, inicializar_particula

def ejecutar_pso(funcion_config, param_config, num_particulas=30, max_iter=100):
    dim = funcion_config["dim"]
    bounds = funcion_config["bounds"]
    w = param_config["w"]
    c1 = param_config["c1"]
    c2 = param_config["c2"]

    # Inicialización
    posiciones = np.array([inicializar_particula(bounds, dim) for _ in range(num_particulas)])
    velocidades = np.random.uniform(-1, 1, (num_particulas, dim))
    pbest = posiciones.copy()
    pbest_val = np.array([evaluar_funcion(funcion_config, x) for x in posiciones])
    gbest_idx = np.argmin(pbest_val)
    gbest = pbest[gbest_idx].copy()
    gbest_val = pbest_val[gbest_idx]
    historial = [gbest_val]

    for iter in range(max_iter):
        for i in range(num_particulas):
            valor = evaluar_funcion(funcion_config, posiciones[i])
            if valor < pbest_val[i]:
                pbest[i] = posiciones[i].copy()
                pbest_val[i] = valor
            if valor < gbest_val:
                gbest = posiciones[i].copy()
                gbest_val = valor

        # Actualizar velocidades y posiciones
        r1 = np.random.rand(num_particulas, dim)
        r2 = np.random.rand(num_particulas, dim)
        velocidades = (
            w * velocidades
            + c1 * r1 * (pbest - posiciones)
            + c2 * r2 * (gbest - posiciones)
        )
        posiciones += velocidades

        # Aplicar límites
        posiciones = np.clip(posiciones, bounds[0], bounds[1])

        historial.append(gbest_val)

    return gbest, gbest_val, historial