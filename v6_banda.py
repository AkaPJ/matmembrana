"""
Versión 6: Almacenamiento en banda — dgbmv
La matriz M tiene kl=N subdiagonales y ku=N superdiagonales.
En vez de almacenar N⁴ elementos (la mayoría ceros), se guarda
en formato banda compacto de (kl+ku+1) × N² = (2N+1) × N².
La rutina DGBMV de BLAS opera directamente sobre este formato.
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


def creasis_banda(N, mu):
    """Construye la matriz M directamente en formato banda LAPACK.

    Formato: array AB de (kl+ku+1) filas × N² columnas
    donde AB[ku + i - j, j] = M[i, j]

    Para nuestra matriz: kl = ku = N
    AB tiene 2N+1 filas × N² columnas
    """
    n2 = N * N
    kl = N
    ku = N
    AB = np.zeros((kl + ku + 1, n2), dtype=np.float64, order='F')

    # Diagonal principal (k=0): fila ku
    AB[ku, :] = 1 + 4 * mu

    # Diagonal +1 (k=+1): fila ku-1
    diag_1 = -np.ones(n2 - 1) * mu
    for i in range(0, n2 - 1, N):
        diag_1[i] = 0
    AB[ku - 1, 1:] = diag_1

    # Diagonal -1 (k=-1): fila ku+1
    AB[ku + 1, :n2 - 1] = diag_1

    # Diagonal +N (k=+N): fila ku-N = fila 0
    AB[ku - N, N:] = -np.ones(n2 - N) * mu

    # Diagonal -N (k=-N): fila ku+N = fila 2N
    AB[ku + N, :n2 - N] = -np.ones(n2 - N) * mu

    return AB, kl, ku


N = 20
n2 = N * N

# Crear ambas versiones para verificar
M = creasis(N, -1)
AB, kl, ku = creasis_banda(N, -1)
b = np.random.rand(n2)

# Verificar que da el mismo resultado
x_ref = M @ b
x_banda = blas.dgbmv(m=n2, n=n2, kl=kl, ku=ku, alpha=1.0, a=AB, x=b)
print(f"Verificación banda vs densa: {np.linalg.norm(x_ref - x_banda):.2e}")
print(f"Tamaño densa:  {n2}×{n2} = {n2 * n2} elementos")
print(f"Tamaño banda:  {2 * N + 1}×{n2} = {(2 * N + 1) * n2} elementos ({100 * (2 * N + 1) * n2 / (n2 * n2):.1f}%)\n")

start_time = time.perf_counter()
for vuelta in range(50):
    x0 = blas.dgbmv(m=n2, n=n2, kl=kl, ku=ku, alpha=1.0, a=AB, x=b)
end_time = time.perf_counter()

t = end_time - start_time
print(f"BLAS dgbmv: {t:.6f} s")
error = np.linalg.norm(x_ref - x0)
print(f"Error: {error:.2e}")