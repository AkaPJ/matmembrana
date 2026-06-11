"""
info_sistema.py — Muestra información del hardware del sistema
Ejecutar en el PC de la uni antes de lanzar el benchmark
"""
import platform
import subprocess
import sys

print("=" * 60)
print("INFORMACIÓN DEL SISTEMA")
print("=" * 60)

# --- Sistema operativo ---
print(f"\nOS:        {platform.system()} {platform.release()}")
print(f"Máquina:   {platform.machine()}")
print(f"Python:    {sys.version.split()[0]}")

# --- CPU ---
print("\n--- CPU ---")
print(f"Procesador: {platform.processor()}")

try:
    import psutil
    print(f"Cores físicos:  {psutil.cpu_count(logical=False)}")
    print(f"Cores lógicos:  {psutil.cpu_count(logical=True)}")
    freq = psutil.cpu_freq()
    if freq:
        print(f"Frecuencia:     {freq.max:.0f} MHz")
    ram = psutil.virtual_memory()
    print(f"RAM total:      {ram.total / 1e9:.1f} GB")
except ImportError:
    print("(instala psutil para más detalles: pip install psutil)")

# Info CPU desde /proc/cpuinfo en Linux
try:
    with open('/proc/cpuinfo') as f:
        for line in f:
            if 'model name' in line:
                print(f"Modelo CPU:     {line.split(':')[1].strip()}")
                break
except FileNotFoundError:
    pass  # Windows o Mac

# --- GPU con NumPy/CuPy ---
print("\n--- GPU ---")
try:
    import cupy as cp
    n_gpus = cp.cuda.runtime.getDeviceCount()
    print(f"GPUs detectadas: {n_gpus}")
    for i in range(n_gpus):
        props = cp.cuda.runtime.getDeviceProperties(i)
        nombre = props['name'].decode()
        vram = props['totalGlobalMem'] / 1e9
        sm = props['multiProcessorCount']
        clock = props['clockRate'] / 1e6
        print(f"\n  GPU {i}: {nombre}")
        print(f"  VRAM:          {vram:.1f} GB")
        print(f"  Stream Multiprocessors: {sm}")
        print(f"  Clock:         {clock:.2f} GHz")
        print(f"  CUDA Compute:  {props['major']}.{props['minor']}")
except ImportError:
    print("CuPy no instalado — no se puede detectar GPU NVIDIA")
    # Intentar con nvidia-smi
    try:
        result = subprocess.run(['nvidia-smi',
                                 '--query-gpu=name,memory.total,driver_version',
                                 '--format=csv,noheader'],
                                capture_output=True, text=True)
        if result.returncode == 0:
            print(f"nvidia-smi: {result.stdout.strip()}")
        else:
            print("nvidia-smi no disponible")
    except FileNotFoundError:
        print("nvidia-smi no encontrado")

# --- NumPy y BLAS ---
print("\n--- NumPy / BLAS ---")
import numpy as np
print(f"NumPy:     {np.__version__}")
np.show_config()

print("\n" + "=" * 60)
