"""
Versión 7: Optimización interprocedural / inline
En C, la opción -ipo del compilador Intel permite optimizar a través
de los límites de funciones (inlining). En Python, el equivalente es
codificar la operación directamente en Numba sin pasar por una
multiplicación matricial genérica.

En vez de construir la matriz M y multiplicar, aplicamos el stencil
de 5 puntos directamente: cada punto solo interactúa con sus 4 vecinos
(arriba, abajo, izquierda, derecha) y consigo mismo.

Ventajas:
  - No se almacena la matriz M (ahorro de memoria total)
  - No hay acceso indirecto (como en CSR sparse)
  - Solo 5 operaciones por punto en vez de N² (bucle) o nnz (sparse)
  - Numba compila e inlinea todo a código máquina nativo
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
def matvec_stencil(b, N, mu):
    """Aplica el operador de membrana directamente sin construir M.
    Cada x0[i] se calcula con solo 5 operaciones (el punto + 4 vecinos),
    en vez de recorrer las N² columnas de la fila i."""
    n2 = N * N
    x0 = np.zeros(n2)

    for idx in range(n2):
        # Diagonal principal: siempre
        val = (1.0 + 4.0 * mu) * b[idx]

        # Vecino izquierda (diagonal -1): si no es borde izquierdo
        if idx % N != 0:
            val += (-mu) * b[idx - 1]

        # Vecino derecha (diagonal +1): si no es borde derecho
        if idx % N != (N - 1):
            val += (-mu) * b[idx + 1]

        # Vecino arriba (diagonal -N): si no es primera fila
        if idx >= N:
            val += (-mu) * b[idx - N]

        # Vecino abajo (diagonal +N): si no es última fila
        if idx < n2 - N:
            val += (-mu) * b[idx + N]

        x0[idx] = val

    return x0

N = 20
M = creasis(N, -1)
mu = -1.0
b = np.random.rand(N * N)

# Verificar que da el mismo resultado
x_ref = M @ b
x_stencil = matvec_stencil(b, N, mu)
print(f"Verificación stencil vs densa: {np.linalg.norm(x_ref - x_stencil):.2e}")

# Warm-up (primera llamada compila)
matvec_stencil(b, N, mu)

start_time = time.perf_counter()
for vuelta in range(50):
    x0 = matvec_stencil(b, N, mu)
end_time = time.perf_counter()

t = end_time - start_time
print(f"Stencil inline: {t:.6f} s")
error = np.linalg.norm(x_ref - x0)
print(f"Error: {error:.2e}")