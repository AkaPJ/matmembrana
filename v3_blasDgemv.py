"""
Versión 3: BLAS nivel 2 — dgemv explícito
Llamada directa a la rutina DGEMV de BLAS:
    y = alpha * A @ x + beta * y
Requiere matrices en orden Fortran (column-major) para máxima eficiencia.
"""
import numpy as np
import time
from scipy.linalg import blas

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

N = 20
M = creasis(N, -1)
# Convertir a orden Fortran (column-major) para que BLAS no copie internamente
M_f = np.asfortranarray(M)
b = np.random.rand(N * N)  # dgemv necesita vector 1D

start_time = time.perf_counter()
for vuelta in range(50):
    x0 = blas.dgemv(alpha=1.0, a=M_f, x=b)
end_time = time.perf_counter()

t = end_time - start_time
print(f"BLAS dgemv: {t:.6f} s")
error = np.linalg.norm(M @ b - x0)
print(f"Error: {error:.2e}")