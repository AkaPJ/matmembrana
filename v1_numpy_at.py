"""
Versión 1: NumPy operador @
Usa internamente BLAS nivel 2 (DGEMV) a través de MKL/OpenBLAS.
"""
import numpy as np
import time

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
b = np.random.rand(N * N, 1)

start_time = time.perf_counter()
for vuelta in range(50):
    x0 = M @ b
end_time = time.perf_counter()

t = end_time - start_time
print(f"NumPy M@b: {t:.6f} s")
error = np.linalg.norm(M @ b - x0)
print(f"Error: {error:.2e}")