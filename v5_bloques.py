"""
Versión 5: Algoritmo a bloques (cache blocking)
Divide la operación en bloques de tamaño tb×tb que caben en
la caché L1/L2, minimizando el tráfico entre niveles de memoria.
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

def matvec_blocked(M, b, fil, col, tb):
    x0 = np.zeros(fil)
    for ib in range(0, fil, tb):
        ie = min(ib + tb, fil)
        for jb in range(0, col, tb):
            je = min(jb + tb, col)
            x0[ib:ie] += M[ib:ie, jb:je] @ b[jb:je]
    return x0

N = 20
M = creasis(N, -1)
fil, col = M.shape
b = np.random.rand(N * N)

# Probar varios tamaños de bloque
for tb in [8, 16, 32, 50, 100]:
    start_time = time.perf_counter()
    for vuelta in range(50):
        x0 = matvec_blocked(M, b, fil, col, tb)
    end_time = time.perf_counter()

    t = end_time - start_time
    error = np.linalg.norm(M @ b - x0)
    print(f"Bloques tb={tb:>3d}: {t:.6f} s  |  Error: {error:.2e}")