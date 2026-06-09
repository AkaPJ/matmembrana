"""
Versión 8: CuPy — NumPy en GPU
Transfiere la matriz M y el vector b a la GPU y ejecuta el producto
allí. CuPy usa internamente cuBLAS (BLAS optimizado para NVIDIA).

REQUISITOS: GPU NVIDIA + pip install cupy-cuda12x
(Ejecutar en PC con GTX 1050 Ti o Google Colab con GPU)
"""
import numpy as np
import time

try:
    import cupy as cp
except ImportError:
    print("CuPy no instalado. Ejecutar: pip install cupy-cuda12x")
    print("Requiere GPU NVIDIA (no funciona en Mac M4)")
    exit(1)

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
M_cpu = creasis(N, -1)
b_cpu = np.random.rand(N * N)

# Transferir datos CPU → GPU
M_gpu = cp.asarray(M_cpu)
b_gpu = cp.asarray(b_cpu)

# Warm-up (primera operación en GPU es lenta por inicialización)
_ = M_gpu @ b_gpu
cp.cuda.Stream.null.synchronize()

# Producto en GPU
start = time.perf_counter()
for vuelta in range(50):
    x_gpu = M_gpu @ b_gpu
cp.cuda.Stream.null.synchronize()
end = time.perf_counter()

# Traer resultado GPU → CPU para verificar
x0 = cp.asnumpy(x_gpu)

t = end - start
x_ref = M_cpu @ b_cpu
print(f"CuPy M@b (GPU): {t:.6f} s")
print(f"Error: {np.linalg.norm(x_ref - x0):.2e}")
print(f"GPU: {cp.cuda.runtime.getDeviceProperties(0)['name'].decode()}")
