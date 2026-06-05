"""
Versión 2: scipy.sparse (Sparse BLAS)
Construye M directamente en formato CSR y opera solo con
los ~1918 elementos no nulos en vez de los 160000.
"""
import numpy as np
import time
from scipy.sparse import diags

def creasis_sparse(N, mu):
    n2 = N * N
    diag_main = np.ones(n2) * (1 + 4 * mu)
    diag_1 = -np.ones(n2 - 1) * mu
    for i in range(0, n2 - 1, N):
        diag_1[i] = 0
    diag_N = -np.ones(n2 - N) * mu
    M_sp = diags(
        [diag_N, diag_1, diag_main, diag_1, diag_N],
        offsets=[-N, -1, 0, 1, N],
        shape=(n2, n2),
        format='csr'
    )
    return M_sp

N = 20
M_sp = creasis_sparse(N, -1)
b = np.random.rand(N * N, 1)

print(f"Elementos no nulos: {M_sp.nnz} de {N**4} ({100*M_sp.nnz/N**4:.1f}%)")

start_time = time.perf_counter()
for vuelta in range(50):
    x0 = M_sp @ b
end_time = time.perf_counter()

t = end_time - start_time
print(f"Sparse M@b: {t:.6f} s")
error = np.linalg.norm(M_sp @ b - x0)
print(f"Error: {error:.2e}")