"""
Versión 1b: Operaciones vectorizadas NumPy
Paso intermedio entre el bucle doble y M@b.
Se elimina el bucle interno (j) usando operaciones vectorizadas,
manteniendo el bucle externo (i) sobre las filas.

Tres variantes:
  a) np.dot por fila    — elimina el bucle j con producto escalar
  b) M * b broadcasting — multiplicación elemento a elemento + suma
  c) np.einsum          — notación Einstein, compacta y eficiente
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
fil, col = M.shape
b = np.random.rand(N * N, 1)
b_1d = b.ravel()
x_ref = M @ b

# =============================================
# a) np.dot por fila: elimina el bucle j
# =============================================
start = time.perf_counter()
for vuelta in range(50):
    x0 = np.zeros(fil)
    for i in range(fil):
        x0[i] = np.dot(M[i, :], b_1d)
end = time.perf_counter()
t_dot = end - start
print(f"np.dot por fila:   {t_dot:.6f} s  |  Error: {np.linalg.norm(x_ref.ravel() - x0):.2e}")

# =============================================
# b) M * b broadcasting + np.sum
#    Equivalente a M .* v en MATLAB
# =============================================
start = time.perf_counter()
for vuelta in range(50):
    x0 = np.sum(M * b_1d, axis=1)
end = time.perf_counter()
t_broadcast = end - start
print(f"M * b broadcast:   {t_broadcast:.6f} s  |  Error: {np.linalg.norm(x_ref.ravel() - x0):.2e}")

# =============================================
# c) np.einsum('ij,j->i', M, b)
# =============================================
start = time.perf_counter()
for vuelta in range(50):
    x0 = np.einsum('ij,j->i', M, b_1d)
end = time.perf_counter()
t_einsum = end - start
print(f"np.einsum:         {t_einsum:.6f} s  |  Error: {np.linalg.norm(x_ref.ravel() - x0):.2e}")

# =============================================
# Referencia: M @ b
# =============================================
start = time.perf_counter()
for vuelta in range(50):
    x0 = M @ b
end = time.perf_counter()
t_at = end - start
print(f"M @ b:             {t_at:.6f} s  |  Error: {np.linalg.norm(x_ref - x0):.2e}")