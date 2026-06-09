"""
Versión 12: Numba @cuda.jit — kernel CUDA element-wise
Escribe el kernel CUDA directamente en Python usando el decorador
@cuda.jit de Numba. Cada hilo calcula un elemento temp[idx] = M[idx] * b[j].

Es equivalente a escribir un kernel CUDA en C, pero con sintaxis Python.
Usa las variables cuda.grid, cuda.threadIdx, cuda.blockIdx, cuda.blockDim
que corresponden a las de CUDA.

REQUISITOS: GPU NVIDIA + pip install numba-cuda
(Ejecutar en PC con GTX 1050 Ti o Google Colab con GPU)
"""
import numpy as np
import time

try:
    from numba import cuda
    import numba
except ImportError:
    print("Numba no instalado. Ejecutar: pip install numba numba-cuda")
    exit(1)

if not cuda.is_available():
    print("No se detecta GPU NVIDIA. Ejecutar en PC con GPU o Google Colab.")
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
# Kernel CUDA element-wise con Numba
# Cada hilo calcula un elemento: temp[idx] = M[idx] * b[j]
# =============================================
@cuda.jit
def elementwise_mul_kernel(M_flat, b, temp, n):
    """Kernel element-wise: cada hilo procesa un elemento de M."""
    idx = cuda.grid(1)  # índice global del hilo
    if idx < n * n:
        j = idx % n      # columna
        temp[idx] = M_flat[idx] * b[j]

N = 20
n2 = N * N
M_cpu = creasis(N, -1)
b_cpu = np.random.rand(n2)
x_ref = M_cpu @ b_cpu

# Aplanar M para enviarlo como vector 1D a la GPU
M_flat = M_cpu.ravel()

# Enviar datos a la GPU
M_gpu = cuda.to_device(M_flat)
b_gpu = cuda.to_device(b_cpu)
temp_gpu = cuda.device_array(n2 * n2)  # array temporal en GPU

# Configuración de lanzamiento
threads_per_block = 256
total_elements = n2 * n2
blocks_per_grid = (total_elements + threads_per_block - 1) // threads_per_block

# Warm-up
elementwise_mul_kernel[blocks_per_grid, threads_per_block](M_gpu, b_gpu, temp_gpu, n2)
cuda.synchronize()

# Benchmark
start = time.perf_counter()
for vuelta in range(50):
    elementwise_mul_kernel[blocks_per_grid, threads_per_block](M_gpu, b_gpu, temp_gpu, n2)
    # Traer a CPU para sumar filas (o usar CuPy para sumar en GPU)
    temp_cpu = temp_gpu.copy_to_host().reshape(n2, n2)
    x0 = np.sum(temp_cpu, axis=1)
cuda.synchronize()
end = time.perf_counter()

t = end - start
error = np.linalg.norm(x_ref - x0)
print(f"Numba @cuda.jit element-wise: {t:.6f} s")
print(f"Error: {error:.2e}")
print(f"GPU: {cuda.get_current_device().name}")
