"""
Versión 4: Numba JIT — vectorización automática
El compilador JIT traduce el bucle Python a código máquina con
instrucciones SIMD (SSE/AVX). Equivalente a compilar C con -O3 -mavx.
"""
import numpy as np
import time
from numba import njit

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
    x0 = np.zeros(fil)
    for i in range(fil):
        acc = 0.0
        for j in range(col):
            acc += M[i, j] * b[j]
        x0[i] = acc
    return x0

N = 20
M = creasis(N, -1)
fil, col = M.shape
b = np.random.rand(N * N)

# Warm-up: la primera llamada compila, NO se mide
matvec_numba(M, b, fil, col)

start_time = time.perf_counter()
for vuelta in range(50):
    x0 = matvec_numba(M, b, fil, col)
end_time = time.perf_counter()

t = end_time - start_time
print(f"Numba JIT: {t:.6f} s")
error = np.linalg.norm(M @ b - x0)
print(f"Error: {error:.2e}")