#!/bin/bash

echo "Setting CUDA environment..."

# CUDA (ajustado a tu sistema real)
export CUDA_HOME=/usr/lib/cuda
export CUDA_PATH=/usr/lib/cuda
export PATH=$CUDA_PATH/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_PATH/lib:$LD_LIBRARY_PATH

# (opcional pero útil)
export CUPY_CACHE_DIR=$PWD/.cupy_cache

echo "CUDA ready"
