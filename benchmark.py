"""
benchmark.py — Ejecuta todas las versiones del producto matriz-vector
y muestra una tabla comparativa de tiempos, speedup y errores.
"""
import numpy as np
import time
from scipy.sparse import diags
from scipy.linalg import blas
from multiprocessing import Pool
import multiprocessing

try:
    from numba import njit, prange
    NUMBA_DISPONIBLE = True
except ImportError:
    NUMBA_DISPONIBLE = False
    print("⚠ Numba no instalado (pip install numba). Se omiten versiones Numba.\n")

try:
    from numba import cuda as numba_cuda
    NUMBA_CUDA_DISPONIBLE = NUMBA_DISPONIBLE and numba_cuda.is_available()
except (ImportError, Exception):
    NUMBA_CUDA_DISPONIBLE = False

try:
    import cupy as cp
    CUPY_DISPONIBLE = True
except ImportError:
    CUPY_DISPONIBLE = False
    print("⚠ CuPy no instalado (pip install cupy-cuda12x). Se omiten versiones GPU.\n")

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
    return diags([diag_N, diag_1, diag_main, diag_1, diag_N],
                 offsets=[-N, -1, 0, 1, N], shape=(n2, n2), format='csr')

def creasis_banda(N, mu):
    n2 = N * N
    kl, ku = N, N
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
# Funciones auxiliares
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

    @njit(parallel=True, cache=True)
    def matvec_numba_par(M, b, fil, col):
        x0 = np.zeros(fil)
        for i in prange(fil):
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

    @njit(parallel=True, cache=True)
    def matvec_stencil_par(b, N, mu):
        n2 = N * N
        x0 = np.zeros(n2)
        for idx in prange(n2):
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

if CUPY_DISPONIBLE:
    ewkernel_mul = cp.ElementwiseKernel(
        'float64 m, float64 b', 'float64 temp',
        'temp = m * b', 'ewkernel_mul'
    )

if NUMBA_CUDA_DISPONIBLE:
    @numba_cuda.jit
    def numba_ew_kernel(M_flat, b, temp, n):
        idx = numba_cuda.grid(1)
        if idx < n * n:
            j = idx % n
            temp[idx] = M_flat[idx] * b[j]

# Multiprocessing
M_global = None
b_global = None

def init_worker(M, b):
    global M_global, b_global
    M_global = M
    b_global = b

def matvec_chunk(args):
    ini, fin = args
    return M_global[ini:fin, :] @ b_global

# =============================================
# MAIN
# =============================================
if __name__ == '__main__':
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
    if CUPY_DISPONIBLE:
        print(f"GPU: {cp.cuda.runtime.getDeviceProperties(0)['name'].decode()}")
    print("=" * 65)

    resultados = []

    # V0: Bucle doble
    start = time.perf_counter()
    for vuelta in range(VUELTAS):
        x0 = np.zeros([n2, 1])
        for i in range(fil):
            for j in range(col):
                x0[i] = x0[i] + M[i, j] * b[j]
    t = time.perf_counter() - start
    resultados.append(("Bucle doble", t, np.linalg.norm(x_ref - x0)))

    # V1a: np.dot por fila
    start = time.perf_counter()
    for vuelta in range(VUELTAS):
        x1a = np.zeros(fil)
        for i in range(fil):
            x1a[i] = np.dot(M[i, :], b_1d)
    t = time.perf_counter() - start
    resultados.append(("np.dot por fila", t, np.linalg.norm(x_ref.ravel() - x1a)))

    # V1b: M * b broadcasting
    start = time.perf_counter()
    for vuelta in range(VUELTAS):
        x1b = np.sum(M * b_1d, axis=1)
    t = time.perf_counter() - start
    resultados.append(("M*b broadcast", t, np.linalg.norm(x_ref.ravel() - x1b)))

    # V1c: np.einsum
    start = time.perf_counter()
    for vuelta in range(VUELTAS):
        x1c = np.einsum('ij,j->i', M, b_1d)
    t = time.perf_counter() - start
    resultados.append(("np.einsum", t, np.linalg.norm(x_ref.ravel() - x1c)))

    # V1: NumPy @
    start = time.perf_counter()
    for vuelta in range(VUELTAS):
        x1 = M @ b
    t = time.perf_counter() - start
    resultados.append(("NumPy M@b", t, np.linalg.norm(x_ref - x1)))

    # V2: Sparse
    start = time.perf_counter()
    for vuelta in range(VUELTAS):
        x2 = M_sp @ b
    t = time.perf_counter() - start
    resultados.append(("Sparse CSR", t, np.linalg.norm(x_ref - x2)))

    # V3: BLAS dgemv
    start = time.perf_counter()
    for vuelta in range(VUELTAS):
        x3 = blas.dgemv(alpha=1.0, a=M_f, x=b_1d)
    t = time.perf_counter() - start
    resultados.append(("BLAS dgemv", t, np.linalg.norm(x_ref.ravel() - x3)))

    # V4a: Numba JIT secuencial
    if NUMBA_DISPONIBLE:
        matvec_numba(M, b_1d, fil, col)
        start = time.perf_counter()
        for vuelta in range(VUELTAS):
            x4 = matvec_numba(M, b_1d, fil, col)
        t = time.perf_counter() - start
        resultados.append(("Numba seq", t, np.linalg.norm(x_ref.ravel() - x4)))

    # V4b: Numba JIT paralelo (prange)
    if NUMBA_DISPONIBLE:
        matvec_numba_par(M, b_1d, fil, col)
        start = time.perf_counter()
        for vuelta in range(VUELTAS):
            x4p = matvec_numba_par(M, b_1d, fil, col)
        t = time.perf_counter() - start
        resultados.append(("Numba parallel", t, np.linalg.norm(x_ref.ravel() - x4p)))

    # V5: Bloques
    start = time.perf_counter()
    for vuelta in range(VUELTAS):
        x5 = matvec_blocked(M, b_1d, fil, col, TB)
    t = time.perf_counter() - start
    resultados.append((f"Bloques tb={TB}", t, np.linalg.norm(x_ref.ravel() - x5)))

    # V6: Banda dgbmv
    start = time.perf_counter()
    for vuelta in range(VUELTAS):
        x6 = blas.dgbmv(m=n2, n=n2, kl=kl, ku=ku, alpha=1.0, a=AB, x=b_1d)
    t = time.perf_counter() - start
    resultados.append(("Banda dgbmv", t, np.linalg.norm(x_ref.ravel() - x6)))

    # V7a: Stencil inline secuencial
    if NUMBA_DISPONIBLE:
        matvec_stencil(b_1d, N, mu)
        start = time.perf_counter()
        for vuelta in range(VUELTAS):
            x7 = matvec_stencil(b_1d, N, mu)
        t = time.perf_counter() - start
        resultados.append(("Stencil seq", t, np.linalg.norm(x_ref.ravel() - x7)))

    # V7b: Stencil inline paralelo
    if NUMBA_DISPONIBLE:
        matvec_stencil_par(b_1d, N, mu)
        start = time.perf_counter()
        for vuelta in range(VUELTAS):
            x7p = matvec_stencil_par(b_1d, N, mu)
        t = time.perf_counter() - start
        resultados.append(("Stencil parallel", t, np.linalg.norm(x_ref.ravel() - x7p)))

    # V8: CuPy GPU M@b
    if CUPY_DISPONIBLE:
        M_gpu = cp.asarray(M)
        b_gpu = cp.asarray(b_1d)
        _ = M_gpu @ b_gpu
        cp.cuda.Stream.null.synchronize()
        start = time.perf_counter()
        for vuelta in range(VUELTAS):
            x8_gpu = M_gpu @ b_gpu
        cp.cuda.Stream.null.synchronize()
        t = time.perf_counter() - start
        resultados.append(("CuPy GPU M@b", t, np.linalg.norm(x_ref.ravel() - cp.asnumpy(x8_gpu))))

    # V9: CuPy ElementwiseKernel
    if CUPY_DISPONIBLE:
        b_exp = cp.tile(b_gpu, (n2, 1))
        _ = ewkernel_mul(M_gpu, b_exp)
        cp.cuda.Device().synchronize()
        start = time.perf_counter()
        for vuelta in range(VUELTAS):
            temp_ew = ewkernel_mul(M_gpu, b_exp)
            x9_gpu = cp.sum(temp_ew, axis=1)
        cp.cuda.Device().synchronize()
        t = time.perf_counter() - start
        resultados.append(("ElementwiseKern", t, np.linalg.norm(x_ref.ravel() - cp.asnumpy(x9_gpu))))

    # V10: multiprocessing Pool
    num_proc = multiprocessing.cpu_count()
    trozo = fil // num_proc
    rangos = [(i * trozo, (i + 1) * trozo if i < num_proc - 1 else fil) for i in range(num_proc)]
    start = time.perf_counter()
    for vuelta in range(VUELTAS):
        with Pool(num_proc, initializer=init_worker, initargs=(M, b_1d)) as p:
            res = p.map(matvec_chunk, rangos)
        x10 = np.concatenate(res)
    t = time.perf_counter() - start
    resultados.append((f"Pool ({num_proc} proc)", t, np.linalg.norm(x_ref.ravel() - x10)))

    # V11: Numba @cuda.jit
    if NUMBA_CUDA_DISPONIBLE:
        M_flat_gpu = numba_cuda.to_device(M.ravel())
        b_nc = numba_cuda.to_device(b_1d)
        temp_nc = numba_cuda.device_array(n2 * n2)
        tpb = 256
        bpg = (n2 * n2 + tpb - 1) // tpb
        numba_ew_kernel[bpg, tpb](M_flat_gpu, b_nc, temp_nc, n2)
        numba_cuda.synchronize()
        start = time.perf_counter()
        for vuelta in range(VUELTAS):
            numba_ew_kernel[bpg, tpb](M_flat_gpu, b_nc, temp_nc, n2)
            x11 = temp_nc.copy_to_host().reshape(n2, n2).sum(axis=1)
        numba_cuda.synchronize()
        t = time.perf_counter() - start
        resultados.append(("Numba @cuda.jit", t, np.linalg.norm(x_ref.ravel() - x11)))

    # =============================================
    # Tabla de resultados
    # =============================================
    t_ref = resultados[0][1]
    print(f"\n{'Versión':<18} {'Tiempo (s)':>12} {'Speedup':>10} {'Error':>12}")
    print("-" * 55)
    for nombre, t, e in resultados:
        speedup = t_ref / t
        print(f"{nombre:<18} {t:>12.6f} {speedup:>9.1f}x {e:>12.2e}")
