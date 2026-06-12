import matplotlib.pyplot as plt
import numpy as np

# Versiones finales (solo una v1)
versiones = [
    "v0 – Bucle doble",
    "v1 – NumPy M@b",
    "v2 – CSR sparse",
    "v3 – BLAS dgemv",
    "v4 – Numba seq",
    "v4p – Numba parallel",
    "v5 – Bloques tb=32",
    "v6 – Banda dgbmv",
    "v7 – Stencil seq",
    "v7p – Stencil parallel",
    "v8 – CuPy M@b",
    "v9 – ElementwiseKernel",
    "v10 – Pool (32p)",
    "v11 – Numba cuda.jit"
]

# Speedups correspondientes (orden exacto)
speedups = [
    1.0,        # v0
    3175.5,     # v1
    32433.3,    # v2
    15004.4,    # v3
    2824.7,     # v4
    3016.5,     # v4p
    670.7,      # v5
    37225.7,    # v6
    106372.8,   # v7
    7633.7,     # v7p
    15671.3,    # v8
    582.1,      # v9
    2.0,        # v10
    398.3       # v11
]

# Orden descendente por speedup (el más rápido arriba)
orden = np.argsort(speedups)[::-1]
versiones_ord = [versiones[i] for i in orden]
speedups_ord = [speedups[i] for i in orden]

plt.figure(figsize=(10, 8))
bars = plt.barh(versiones_ord, speedups_ord, color='steelblue')
plt.xscale('log')
plt.xlabel('Speedup (escala logarítmica)', fontsize=12)
plt.title('Speedup respecto al bucle doble original (v0) para N=20', fontsize=14)
plt.grid(axis='x', linestyle='--', alpha=0.7)

# Etiquetas con el valor numérico
for bar in bars:
    width = bar.get_width()
    plt.text(width * 1.05, bar.get_y() + bar.get_height()/2,
             f'{width:.0f}×', va='center', ha='left', fontsize=8)

plt.tight_layout()
plt.savefig('fig_speedup_N20.pdf', bbox_inches='tight')
plt.show()
