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

Para cada uno de estos casos se debe encontrar el mínimo de la función. Se debe elegir una metaheurística que los encuentre realizando 10 iteraciones para cada función utilizando 5 donfiguraciones distintas:

### Particle Swarm Optimization (PSO)
Se elige el presente algoritmo por las siguientes razones:

1. Es adaptable con funciones no lineales y multimodales, semejantes a las propuestas en la tarea. La metaheurística PSO está especialmente diseñada para:
   - Explorar eficazmente el espacio de búsqueda, evitando quedar atrapado prematuramente en óptimos locales
   - Ajustar dinámicamente la explotación y exploración a lo largo de las iteraciones

2. Es sencilla de implementar y conlleva reglas de actualización simples y robustas: PSO se destaca por su simplicidad estructural, en donde:
   - Se requiere solo la actualización de velocidades y posiciones de cada partícula
   - No depende de operadores coimplejos o estructuras adicionales

3. Es versatil y entrega resultados empíricos: PSO demustra un buen rendimiento en la literatura:
   - Utilizado en benchmarks para funciones multimodales
   - Realiza una optimización sin gradiente, que es lo que estamos buscando para este caso
   - No presenta problemas de dimensión críticos

### Configuraciones utilziadas:
- Standard:
	- w: 0.7
	- c1: 1.5
	- c2: 1.5

- High Exploration: 
	- w: 0.9
	- c1: 2.0
	- c2: 1.0

- High Exploitation:
	- w: 0.4
	- c1: 1.0
	- c2: 2.0

- Balanced:
	- w: 0.6
	- c1: 1.7
	- c2: 1.7

- Conservative:
	- w: 0.5
	- c1: 1.0
	- c2: 1.0


