"""
Versión 10: multiprocessing.Pool
Reparte las filas de la matriz M entre los procesos disponibles.
Cada proceso calcula un bloque de filas del resultado x0.
Equivalente Python del parfor/spmd de MATLAB.

Limitación: el GIL de Python impide paralelismo real con threads,
por eso se usan procesos (cada uno con su propia memoria).
El coste de copiar M y b a cada proceso puede ser significativo.
"""
import numpy as np
import time
from multiprocessing import Pool
import multiprocessing

def creasis(N, mu):
    diagonal = np.ones(N * N) * (1 + 4 * mu)
    M = np.diag(diagonal)
    vdi = -np.ones(N * N - 1) * mu
    for i in range(0, N * N - 1, N):
        vdi[i] = 0
    M = M + np.diag(vdi, 1) + np.diag(vdi, -1)
    vdd = -np.ones(N * N - N) * mu
    M = M + np.diag(vdd, -N) + np.diag(vdd, N)
    return M

# Variables globales para que los workers las hereden sin copiar
M_global = None
b_global = None

def init_worker(M, b):
    """Inicializa las variables globales en cada worker."""
    global M_global, b_global
    M_global = M
    b_global = b

def matvec_chunk(args):
    """Calcula un bloque de filas del producto M·b.
    Recibe (ini, fin) y devuelve x0[ini:fin]."""
    ini, fin = args
    return M_global[ini:fin, :] @ b_global

if __name__ == '__main__':
    N = 20
    M = creasis(N, -1)
    fil, col = M.shape
    b = np.random.rand(N * N)
    x_ref = M @ b

    num_procesos = multiprocessing.cpu_count()

    # Crear los rangos de filas para cada proceso
    trozo = fil // num_procesos
    rangos = []
    for i in range(num_procesos):
        ini = i * trozo
        fin = (i + 1) * trozo if i < num_procesos - 1 else fil
        rangos.append((ini, fin))

    # Warm-up del pool
    with Pool(num_procesos, initializer=init_worker, initargs=(M, b)) as p:
        _ = p.map(matvec_chunk, rangos)

    # Benchmark
    start = time.perf_counter()
    for vuelta in range(50):
        with Pool(num_procesos, initializer=init_worker, initargs=(M, b)) as p:
            resultados = p.map(matvec_chunk, rangos)
        x0 = np.concatenate(resultados)
    end = time.perf_counter()

    t = end - start
    error = np.linalg.norm(x_ref - x0)
    print(f"multiprocessing Pool ({num_procesos} procesos): {t:.6f} s")
    print(f"Error: {error:.2e}")
