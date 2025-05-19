import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from funciones import FUNCIONES, evaluar_funcion, inicializar_particula
from pso import ejecutar_pso

def main():
    print("Seleccione la función a analizar:")
    for i, nombre in enumerate(FUNCIONES.keys(), start=1):
        print(f"{i}. {nombre}")

    opcion = int(input("Ingrese el número de la función: "))
    nombre_funcion = list(FUNCIONES.keys())[opcion - 1]
    funcion_config = FUNCIONES[nombre_funcion]

    print(f"\nEjecutando PSO para: {nombre_funcion}")

    # 5 configuraciones de parámetros
    configuraciones = [
        {"w": 0.9, "c1": 2.0, "c2": 2.0},
        {"w": 0.5, "c1": 1.5, "c2": 2.5},
        {"w": 0.7, "c1": 2.5, "c2": 1.5},
        {"w": 0.8, "c1": 2.0, "c2": 1.0},
        {"w": 0.6, "c1": 2.2, "c2": 1.8},
    ]

    resultados = []

    for idx, config in enumerate(configuraciones):
        print(f"\nConfiguración {idx+1}: {config}")
        convergencias = []
        mejores_valores = []

        for ejec in range(10):
            gbest, gbest_val, historial = ejecutar_pso(funcion_config, config)
            convergencias.append(historial)
            mejores_valores.append(gbest_val)

        resultados.append({
            "config": config,
            "mejores_valores": mejores_valores,
            "convergencias": convergencias
        })

    # Mostrar tablas de resultados
    for idx, resultado in enumerate(resultados):
        print(f"\n>>> Configuración {idx+1}: {resultado['config']}")
        print(f"  Mejor valor promedio: {np.mean(resultado['mejores_valores']):.6f}")
        print(f"  Mejor valor mínimo:   {np.min(resultado['mejores_valores']):.6f}")
        print(f"  Desviación estándar:  {np.std(resultado['mejores_valores']):.6f}")

    # Graficar todas las convergencias promedio juntas
    plt.figure()
    sns.set(style="whitegrid")
    for idx, resultado in enumerate(resultados):
        convergencias_avg = np.mean(np.array(resultado['convergencias']), axis=0)
        label = f"Config {idx+1}: w={resultado['config']['w']}, c1={resultado['config']['c1']}, c2={resultado['config']['c2']}"
        plt.plot(convergencias_avg, label=label)
    plt.title(f'Convergencia promedio - {nombre_funcion}')
    plt.xlabel('Iteraciones')
    plt.xlim(0, 20)
    plt.ylabel('Valor de la función')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
