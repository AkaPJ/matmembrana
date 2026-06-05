"""
benchmark.py — Ejecuta todas las versiones del producto matriz-vector
y muestra una tabla comparativa de tiempos, speedup y errores.
"""
import numpy as np
import time
from scipy.sparse import diags
from scipy.linalg import blas

try:
    from numba import njit
    NUMBA_DISPONIBLE = True
except ImportError:
    NUMBA_DISPONIBLE = False
    print("⚠ Numba no instalado (pip install numba). Se omiten versiones Numba.\n")

# =============================================
# Construcción de matrices
# =============================================
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

def creasis_banda(N, mu):
    n2 = N * N
    kl = N
    ku = N
    AB = np.zeros((kl + ku + 1, n2), dtype=np.float64, order='F')
    AB[ku, :] = 1 + 4 * mu
    diag_1 = -np.ones(n2 - 1) * mu
    for i in range(0, n2 - 1, N):
        diag_1[i] = 0
    AB[ku - 1, 1:] = diag_1
    AB[ku + 1, :n2 - 1] = diag_1
    AB[ku - N, N:] = -np.ones(n2 - N) * mu
    AB[ku + N, :n2 - N] = -np.ones(n2 - N) * mu
    return AB, kl, ku

# =============================================
# Funciones de cada versión
# =============================================
def matvec_blocked(M, b, fil, col, tb):
    x0 = np.zeros(fil)
    for ib in range(0, fil, tb):
        ie = min(ib + tb, fil)
        for jb in range(0, col, tb):
            je = min(jb + tb, col)
            x0[ib:ie] += M[ib:ie, jb:je] @ b[jb:je]
    return x0

if NUMBA_DISPONIBLE:
    @njit(cache=True)
    def matvec_numba(M, b, fil, col):
        x0 = np.zeros(fil)
        for i in range(fil):
            acc = 0.0
            for j in range(col):
                acc += M[i, j] * b[j]
            x0[i] = acc
        return x0

    @njit(cache=True)
    def matvec_stencil(b, N, mu):
        n2 = N * N
        x0 = np.zeros(n2)
        for idx in range(n2):
            val = (1.0 + 4.0 * mu) * b[idx]
            if idx % N != 0:
                val += (-mu) * b[idx - 1]
            if idx % N != (N - 1):
                val += (-mu) * b[idx + 1]
            if idx >= N:
                val += (-mu) * b[idx - N]
            if idx < n2 - N:
                val += (-mu) * b[idx + N]
            x0[idx] = val
        return x0

# =============================================
# Configuración
# =============================================
N = 20
VUELTAS = 50
TB = 32
mu = -1.0

M = creasis(N, mu)
M_sp = creasis_sparse(N, mu)
M_f = np.asfortranarray(M)
AB, kl, ku = creasis_banda(N, mu)
fil, col = M.shape
n2 = N * N
b = np.random.rand(n2, 1)
b_1d = b.ravel()

x_ref = M @ b

print(f"N = {N}  →  Matriz {fil}×{col}  |  NNZ = {M_sp.nnz} de {fil*col} ({100*M_sp.nnz/(fil*col):.1f}%)")
print(f"Vueltas = {VUELTAS}  |  Bloque = {TB}")
print("=" * 65)

resultados = []

# =============================================
# V0: Bucle doble
# =============================================
start = time.perf_counter()
for vuelta in range(VUELTAS):
    x0 = np.zeros([n2, 1])
    for i in range(fil):
        for j in range(col):
            x0[i] = x0[i] + M[i, j] * b[j]
t = time.perf_counter() - start
e = np.linalg.norm(x_ref - x0)
resultados.append(("Bucle doble", t, e))

# =============================================
# V1a: np.dot por fila
# =============================================
start = time.perf_counter()
for vuelta in range(VUELTAS):
    x1a = np.zeros(fil)
    for i in range(fil):
        x1a[i] = np.dot(M[i, :], b_1d)
t = time.perf_counter() - start
e = np.linalg.norm(x_ref.ravel() - x1a)
resultados.append(("np.dot por fila", t, e))

# =============================================
# V1b: M * b broadcasting + sum
# =============================================
start = time.perf_counter()
for vuelta in range(VUELTAS):
    x1b = np.sum(M * b_1d, axis=1)
t = time.perf_counter() - start
e = np.linalg.norm(x_ref.ravel() - x1b)
resultados.append(("M*b broadcast", t, e))

# =============================================
# V1c: np.einsum
# =============================================
start = time.perf_counter()
for vuelta in range(VUELTAS):
    x1c = np.einsum('ij,j->i', M, b_1d)
t = time.perf_counter() - start
e = np.linalg.norm(x_ref.ravel() - x1c)
resultados.append(("np.einsum", t, e))

# =============================================
# V1: NumPy @
# =============================================
start = time.perf_counter()
for vuelta in range(VUELTAS):
    x1 = M @ b
t = time.perf_counter() - start
e = np.linalg.norm(x_ref - x1)
resultados.append(("NumPy M@b", t, e))

# =============================================
# V2: Sparse
# =============================================
start = time.perf_counter()
for vuelta in range(VUELTAS):
    x2 = M_sp @ b
t = time.perf_counter() - start
e = np.linalg.norm(x_ref - x2)
resultados.append(("Sparse CSR", t, e))

# =============================================
# V3: BLAS dgemv
# =============================================
start = time.perf_counter()
for vuelta in range(VUELTAS):
    x3 = blas.dgemv(alpha=1.0, a=M_f, x=b_1d)
t = time.perf_counter() - start
e = np.linalg.norm(x_ref.ravel() - x3)
resultados.append(("BLAS dgemv", t, e))

# =============================================
# V4: Numba JIT
# =============================================
if NUMBA_DISPONIBLE:
    matvec_numba(M, b_1d, fil, col)  # warm-up
    start = time.perf_counter()
    for vuelta in range(VUELTAS):
        x4 = matvec_numba(M, b_1d, fil, col)
    t = time.perf_counter() - start
    e = np.linalg.norm(x_ref.ravel() - x4)
    resultados.append(("Numba JIT", t, e))

# =============================================
# V5: Bloques
# =============================================
start = time.perf_counter()
for vuelta in range(VUELTAS):
    x5 = matvec_blocked(M, b_1d, fil, col, TB)
t = time.perf_counter() - start
e = np.linalg.norm(x_ref.ravel() - x5)
resultados.append((f"Bloques tb={TB}", t, e))

# =============================================
# V6: Banda dgbmv
# =============================================
start = time.perf_counter()
for vuelta in range(VUELTAS):
    x6 = blas.dgbmv(m=n2, n=n2, kl=kl, ku=ku, alpha=1.0, a=AB, x=b_1d)
t = time.perf_counter() - start
e = np.linalg.norm(x_ref.ravel() - x6)
resultados.append(("Banda dgbmv", t, e))

# =============================================
# V7: Stencil inline (Numba)
# =============================================
if NUMBA_DISPONIBLE:
    matvec_stencil(b_1d, N, mu)  # warm-up
    start = time.perf_counter()
    for vuelta in range(VUELTAS):
        x7 = matvec_stencil(b_1d, N, mu)
    t = time.perf_counter() - start
    e = np.linalg.norm(x_ref.ravel() - x7)
    resultados.append(("Stencil inline", t, e))

# =============================================
# Tabla de resultados
# =============================================
t_ref = resultados[0][1]

print(f"\n{'Versión':<18} {'Tiempo (s)':>12} {'Speedup':>10} {'Error':>12}")
print("-" * 55)
for nombre, t, e in resultados:
    speedup = t_ref / t
    print(f"{nombre:<18} {t:>12.6f} {speedup:>9.1f}x {e:>12.2e}")