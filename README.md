# Tarea 3 - Funciones Multimodales


La optimización de funciones multimodales representa un desafío considerable, ya que  presentan múltiples óptimos locales. 

Como alternativa, es recomendable emplear metaheurísticas basadas en población, las cuales tienden a ofrecer mejores resultados en este tipo de problemas.  

En particular, para esta tarea se deben considerar las siguientes funciones multimodales definidas en dominios continuos:

$$f_1(x) = 4 - 4x_{1}^{3} - 4x_1 + x_2^{2}, \quad \text{con }-5 \leq x_i \leq 5$$

---

$$f_2(x) = \frac{1}{899}(\sum_{i=1}^{6}x_{i}^{2}2^i - 1745), \quad \text{con }0 \leq x_i \leq 1$$

---

$$f_3(x) = (x_{1}^{6} + x_{2}^{4} + 12)^2 + (2x_1 + x_2 - 4^2), \quad \text{con }-500 \leq x_i \leq 500$$

---

$$f_4(x) = \sum_{i=1}^{10}[(\ln{(x_i-2)})^2 + (\ln{(10-x_i)})^2] - (\prod_{i=1}^{10}x_i)^{0,2}, \quad \text{con }-2001 \leq x_i \leq 10$$

---

Para cada uno de estos casos se debe encontrar el mínimo de la función. En particular se pide lo siguiente:

1. Se diseña e implementa el algoritmo Particle Swarm Optimization (PSO)
2. Para cada una de las funciones se realizan 10 ejecuciones con la metaheurística. Se considerarn 5 configuraciones de parámetros distintas. Muestre sus resultados e incluya gráficos de convergencia de parámetros distintas. Muestre sus resultados e incluya gráficos de convergencia para cada configuración de parámetros.