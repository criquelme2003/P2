#include <cuda_runtime.h>

// Kernel para calcular la suma de una columna
__global__ void columnSumKernel(const float* data, int rows, int col_offset, 
                                float* result) {
    extern __shared__ float shared_sum[];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Cargar datos en memoria compartida
    shared_sum[tid] = (idx < rows) ? data[col_offset + idx] : 0.0f;
    __syncthreads();
    
    // Reducción en memoria compartida
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_sum[tid] += shared_sum[tid + stride];
        }
        __syncthreads();
    }
    
    // El primer thread escribe el resultado
    if (tid == 0) {
        atomicAdd(result, shared_sum[0]);
    }
}

// Kernel para calcular la suma de cuadrados (para desviación estándar)
__global__ void columnSumSquaresKernel(const float* data, int rows, 
                                       int col_offset, float mean, 
                                       float* result) {
    extern __shared__ float shared_sum[];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Calcular (x - mean)^2 y cargar en memoria compartida
    if (idx < rows) {
        float diff = data[col_offset + idx] - mean;
        shared_sum[tid] = diff * diff;
    } else {
        shared_sum[tid] = 0.0f;
    }
    __syncthreads();
    
    // Reducción en memoria compartida
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_sum[tid] += shared_sum[tid + stride];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        atomicAdd(result, shared_sum[0]);
    }
}
