import matplotlib.pyplot as plt
import numpy as np
from multiprocessing import Pool, cpu_count

def logistic_map(r, x):
    x = np.clip(x, 1e-10, 1 - 1e-10)
    return r * x * (1 - x)

def calculate_bifurcation(r):
    x = 1e-5
    iterations = 1000
    last = 100
    # Iteraciones para alcanzar la estabilidad
    for _ in range(iterations):
        x = logistic_map(r, x)
    
    # Recoge los últimos valores después de alcanzar la estabilidad
    result = []
    for _ in range(last):
        x = logistic_map(r, x)
        result.append((r, x))
    
    return result

def main():
    r_values = np.linspace(0.0, 4.0, 1000)  # Permitir r hasta 8.0
    pool = Pool(cpu_count())
    
    # Usar multiprocessing para calcular en paralelo
    results = pool.map(calculate_bifurcation, r_values)
    
    # Flattening the list of results
    x_vals = [x for sublist in results for x in sublist]
    
    r_vals, x_vals = zip(*x_vals)
    
    plt.plot(r_vals, x_vals, ',k', alpha=0.25)
    plt.xlim(0.0, 4.0)
    plt.ylim(0, 1)
    plt.title('Diagrama de Bifurcación del Mapa Logístico.')
    plt.xlabel('r')
    plt.ylabel('x')
    plt.show()

if __name__ == '__main__':
    main()
