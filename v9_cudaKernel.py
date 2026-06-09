"""
Versión 11: CuPy ElementwiseKernel
Equivalente al arrayfun de MATLAB Parallel Toolbox.
Define una operación elemento a elemento que se ejecuta en la GPU,
un hilo por cada elemento de la matriz M.

El kernel multiplica M[i,j] * b[j] para cada posición (i,j),
y luego se suman las filas con cp.sum para obtener x0.

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

# =============================================
# ElementwiseKernel: similar a arrayfun en MATLAB
# Recibe M[i,j] y b_expanded[i,j] (= b[j] replicado)
# Devuelve temp[i,j] = M[i,j] * b_expanded[i,j]
# =============================================
kernel_mul = cp.ElementwiseKernel(
    'float64 m, float64 b',   # argumentos de entrada
    'float64 temp',            # argumento de salida
    'temp = m * b',            # operación elemento a elemento
    'kernel_mul'               # nombre del kernel
)

N = 20
n2 = N * N
M_cpu = creasis(N, -1)
b_cpu = np.random.rand(n2)
x_ref = M_cpu @ b_cpu

# Enviar datos a la GPU
M_gpu = cp.asarray(M_cpu)
b_gpu = cp.asarray(b_cpu)

# Expandir b para que tenga la misma forma que M (broadcasting manual)
# b_expanded[i,j] = b[j] para todo i
b_expanded = cp.tile(b_gpu, (n2, 1))  # shape: (n2, n2)

# Warm-up
temp = kernel_mul(M_gpu, b_expanded)
x0_gpu = cp.sum(temp, axis=1)
cp.cuda.Device().synchronize()

# Benchmark
start = time.perf_counter()
for vuelta in range(50):
    temp = kernel_mul(M_gpu, b_expanded)
    x0_gpu = cp.sum(temp, axis=1)
cp.cuda.Device().synchronize()
end = time.perf_counter()

x0 = cp.asnumpy(x0_gpu)
t = end - start
error = np.linalg.norm(x_ref - x0)
print(f"CuPy ElementwiseKernel: {t:.6f} s")
print(f"Error: {error:.2e}")
print(f"GPU: {cp.cuda.runtime.getDeviceProperties(0)['name'].decode()}")
