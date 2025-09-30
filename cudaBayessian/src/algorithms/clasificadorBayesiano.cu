#include <iostream>
#include <core.cuh>
#include <kernels.cuh>
void clasificarBayesiano(const float *h_test_data,
                         const ColumnStats *h_stats_pos,
                         const ColumnStats *h_stats_neg,
                         float prior_pos,
                         float prior_neg,
                         int *h_predictions,
                         float *h_log_likelihood_pos,
                         float *h_log_likelihood_neg,
                         int n_test,
                         int cols,
                         int target_col_index,
                         float epsilon)
{

    // Calcular log priors
    float log_prior_pos = logf(prior_pos);
    float log_prior_neg = logf(prior_neg);

    // Asignar memoria en GPU
    float *d_test_data;
    ColumnStats *d_stats_pos, *d_stats_neg;
    int *d_predictions;
    float *d_log_likelihood_pos, *d_log_likelihood_neg;

    cudaMalloc(&d_test_data, n_test * cols * sizeof(float));
    cudaMalloc(&d_stats_pos, cols * sizeof(ColumnStats));
    cudaMalloc(&d_stats_neg, cols * sizeof(ColumnStats));
    cudaMalloc(&d_predictions, n_test * sizeof(int));
    cudaMalloc(&d_log_likelihood_pos, n_test * sizeof(float));
    cudaMalloc(&d_log_likelihood_neg, n_test * sizeof(float));

    // Copiar datos a GPU
    cudaMemcpy(d_test_data, h_test_data, n_test * cols * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_stats_pos, h_stats_pos, cols * sizeof(ColumnStats),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_stats_neg, h_stats_neg, cols * sizeof(ColumnStats),
               cudaMemcpyHostToDevice);

    // Configurar kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (n_test + threadsPerBlock - 1) / threadsPerBlock;

    // Ejecutar kernel
    naiveBayesKernel<<<blocksPerGrid, threadsPerBlock>>>(
        d_test_data,
        d_stats_pos,
        d_stats_neg,
        log_prior_pos,
        log_prior_neg,
        d_predictions,
        d_log_likelihood_pos,
        d_log_likelihood_neg,
        n_test,
        cols,
        target_col_index,
        epsilon);

    // Verificar errores
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        std::cerr << "Error en kernel: " << cudaGetErrorString(err) << std::endl;
    }

    cudaDeviceSynchronize();

    // Copiar resultados de vuelta
    cudaMemcpy(h_predictions, d_predictions, n_test * sizeof(int),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(h_log_likelihood_pos, d_log_likelihood_pos, n_test * sizeof(float),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(h_log_likelihood_neg, d_log_likelihood_neg, n_test * sizeof(float),
               cudaMemcpyDeviceToHost);

    // Liberar memoria GPU
    cudaFree(d_test_data);
    cudaFree(d_stats_pos);
    cudaFree(d_stats_neg);
    cudaFree(d_predictions);
    cudaFree(d_log_likelihood_pos);
    cudaFree(d_log_likelihood_neg);
}