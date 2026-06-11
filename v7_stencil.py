"""
Versión 7: Optimización interprocedural / inline

En C, la opción -ipo del compilador Intel permite optimizar a través
de los límites de funciones (inlining). En Python, el equivalente es
codificar la operación directamente en Numba sin pasar por una
multiplicación matricial genérica.

En vez de construir la matriz M y multiplicar, aplicamos el stencil
de 5 puntos directamente: cada punto interactúa con sus 4 vecinos y
consigo mismo.

Ventajas:
  - No se almacena la matriz M (ahorro de memoria total)
  - No hay acceso indirecto (como en CSR sparse)
  - Solo 5 operaciones por punto en vez de N² (bucle) o nnz (sparse)
  - Numba compila e inlinea todo a código máquina nativo

NOTA IMPORTANTE sobre las condiciones de borde:
La función creasis() pone los ceros en vdi[0], vdi[N], vdi[2N]... lo
cual NO se corresponde con los cortes "físicos" entre filas de la
rejilla (que estarían en N-1, 2N-1, ...). El stencil debe reproducir
exactamente la matriz M que genera creasis, no el laplaciano teórico.
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
def matvec_stencil(b, N, mu):
    """Aplica el operador directamente sin construir M.

    Condiciones que reproducen exactamente la matriz creasis:
      - M[idx, idx+1] = -mu si idx % N != 0    (vecino derecha)
      - M[idx, idx-1] = -mu si (idx-1) % N != 0  (vecino izquierda)
      - M[idx, idx-N] = -mu si idx >= N
      - M[idx, idx+N] = -mu si idx + N < N²
    """
    n2 = N * N
    x0 = np.zeros(n2)
    for idx in range(n2):
        # Diagonal principal
        val = (1.0 + 4.0 * mu) * b[idx]
        # Vecino derecha: M[idx, idx+1] = vdi[idx]
        if idx + 1 < n2 and idx % N != 0:
            val += (-mu) * b[idx + 1]
        # Vecino izquierda: M[idx, idx-1] = vdi[idx-1]
        if idx >= 1 and (idx - 1) % N != 0:
            val += (-mu) * b[idx - 1]
        # Vecino arriba (diagonal -N)
        if idx >= N:
            val += (-mu) * b[idx - N]
        # Vecino abajo (diagonal +N)
        if idx + N < n2:
            val += (-mu) * b[idx + N]
        x0[idx] = val
    return x0


@njit(parallel=True, cache=True)
def matvec_stencil_par(b, N, mu):
    """Versión paralela: prange reparte las n2 iteraciones entre cores."""
    n2 = N * N
    x0 = np.zeros(n2)
    for idx in prange(n2):
        val = (1.0 + 4.0 * mu) * b[idx]
        if idx + 1 < n2 and idx % N != 0:
            val += (-mu) * b[idx + 1]
        if idx >= 1 and (idx - 1) % N != 0:
            val += (-mu) * b[idx - 1]
        if idx >= N:
            val += (-mu) * b[idx - N]
        if idx + N < n2:
            val += (-mu) * b[idx + N]
        x0[idx] = val
    return x0


if __name__ == '__main__':
    N = 20
    mu = -1.0
    M = creasis(N, mu)
    b = np.random.rand(N * N)

    # Verificar que el stencil reproduce exactamente M@b
    x_ref = M @ b
    x_stencil = matvec_stencil(b, N, mu)
    x_stencil_par = matvec_stencil_par(b, N, mu)

    err_seq = np.linalg.norm(x_ref - x_stencil)
    err_par = np.linalg.norm(x_ref - x_stencil_par)
    print(f"Verificación stencil seq vs M@b:      {err_seq:.2e}")
    print(f"Verificación stencil parallel vs M@b: {err_par:.2e}")

    if err_seq > 1e-10:
        print("⚠ ERROR: el stencil no coincide con M. Revisa las condiciones.")
    else:
        print("✓ Stencil correcto")

    # Warm-up
    matvec_stencil(b, N, mu)
    matvec_stencil_par(b, N, mu)

    # Medida secuencial
    start_time = time.perf_counter()
    for vuelta in range(50):
        x0 = matvec_stencil(b, N, mu)
    t_seq = time.perf_counter() - start_time
    print(f"Stencil seq      : {t_seq:.6f} s")

    # Medida paralela
    start_time = time.perf_counter()
    for vuelta in range(50):
        x0p = matvec_stencil_par(b, N, mu)
    t_par = time.perf_counter() - start_time
    print(f"Stencil parallel : {t_par:.6f} s")
    print(f"Speedup paralelo : {t_seq / t_par:.2f}x")
