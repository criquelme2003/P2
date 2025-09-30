__global__ void countConfusionMatrixKernel(const int *y_true,
                                           const int *y_pred,
                                           int *TP, int *TN,
                                           int *FP, int *FN,
                                           int n)
{
    extern __shared__ int shared_counts[];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Inicializar memoria compartida: [TP, TN, FP, FN]
    if (tid < 4)
    {
        shared_counts[tid] = 0;
    }
    __syncthreads();

    // Cada thread cuenta su elemento
    if (idx < n)
    {
        int true_val = y_true[idx];
        int pred_val = y_pred[idx];

        if (true_val == 1 && pred_val == 1)
        {
            atomicAdd(&shared_counts[0], 1); // TP
        }
        else if (true_val == 0 && pred_val == 0)
        {
            atomicAdd(&shared_counts[1], 1); // TN
        }
        else if (true_val == 0 && pred_val == 1)
        {
            atomicAdd(&shared_counts[2], 1); // FP
        }
        else if (true_val == 1 && pred_val == 0)
        {
            atomicAdd(&shared_counts[3], 1); // FN
        }
    }
    __syncthreads();

    // El primer thread de cada bloque actualiza los contadores globales
    if (tid < 4)
    {
        atomicAdd(&TP[tid], shared_counts[tid]);
    }
}
