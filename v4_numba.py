"""
Versión 4: Numba JIT — secuencial vs paralelo
a) @njit(cache=True)              → compila a código máquina, 1 core
b) @njit(parallel=True, cache=True) → compila + reparte bucle entre cores (prange)

prange es el equivalente de parfor en MATLAB:
las iteraciones del bucle externo se reparten entre los cores disponibles.
"""
import numpy as np
import time
from numba import njit, prange

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

@njit(cache=True)
def matvec_numba(M, b, fil, col):
    """Versión secuencial: 1 solo core."""
    x0 = np.zeros(fil)
    for i in range(fil):
        acc = 0.0
        for j in range(col):
            acc += M[i, j] * b[j]
        x0[i] = acc
    return x0

@njit(parallel=True, cache=True)
def matvec_numba_parallel(M, b, fil, col):
    """Versión paralela: reparte filas entre cores con prange."""
    x0 = np.zeros(fil)
    for i in prange(fil):       # ← prange: cada core procesa un bloque de filas
        acc = 0.0
        for j in range(col):    # ← range: bucle interno secuencial
            acc += M[i, j] * b[j]
        x0[i] = acc
    return x0

N = 20
M = creasis(N, -1)
fil, col = M.shape
b = np.random.rand(N * N)
x_ref = M @ b

# Warm-up (compilar ambas versiones)
matvec_numba(M, b, fil, col)
matvec_numba_parallel(M, b, fil, col)

# Secuencial
start = time.perf_counter()
for vuelta in range(50):
    x0 = matvec_numba(M, b, fil, col)
end = time.perf_counter()
t_seq = end - start
print(f"Numba secuencial:  {t_seq:.6f} s  |  Error: {np.linalg.norm(x_ref - x0):.2e}")
