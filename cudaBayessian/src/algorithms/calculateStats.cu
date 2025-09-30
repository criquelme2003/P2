#include <core.cuh>
#include <kernels.cuh>

// Función para calcular estadísticas de una columna en GPU
ColumnStats calculateColumnStats(const float *d_data, int rows, int columnIndex)
{
    ColumnStats stats;

    // Configuración del kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (rows + threadsPerBlock - 1) / threadsPerBlock;
    int sharedMemSize = threadsPerBlock * sizeof(float);

    // Offset de la columna (column-major)
    int col_offset = columnIndex * rows;

    // 1. Calcular la media
    float *d_sum;
    cudaMalloc(&d_sum, sizeof(float));
    cudaMemset(d_sum, 0, sizeof(float));

    columnSumKernel<<<blocksPerGrid, threadsPerBlock, sharedMemSize>>>(
        d_data, rows, col_offset, d_sum);

    float sum;
    cudaMemcpy(&sum, d_sum, sizeof(float), cudaMemcpyDeviceToHost);
    stats.mean = sum / rows;

    // 2. Calcular la desviación estándar
    float *d_sum_squares;
    cudaMalloc(&d_sum_squares, sizeof(float));
    cudaMemset(d_sum_squares, 0, sizeof(float));

    columnSumSquaresKernel<<<blocksPerGrid, threadsPerBlock, sharedMemSize>>>(
        d_data, rows, col_offset, stats.mean, d_sum_squares);

    float sum_squares;
    cudaMemcpy(&sum_squares, d_sum_squares, sizeof(float), cudaMemcpyDeviceToHost);
    stats.sd = std::sqrt(sum_squares / (rows - 1)); // Desviación estándar muestral

    // Liberar memoria temporal
    cudaFree(d_sum);
    cudaFree(d_sum_squares);

    return stats;
}

// Función principal para calcular estadísticas de todas las columnas
void calculateAllStats(const float *d_data, int rows, int cols,
                       ColumnStats *stats_array)
{
    // Procesar cada columna secuencialmente (pero cada columna se procesa en paralelo internamente)
    for (int j = 0; j < cols; j++)
    {
        stats_array[j] = calculateColumnStats(d_data, rows, j);
    }
}

// Función wrapper completa que replica el código R
void computeClassStatistics(const float *d_pos_data, int pos_rows,
                            const float *d_neg_data, int neg_rows,
                            int cols,
                            ColumnStats *stats_pos,
                            ColumnStats *stats_neg)
{
    // Calcular estadísticas para clase positiva
    calculateAllStats(d_pos_data, pos_rows, cols, stats_pos);

    // Calcular estadísticas para clase negativa
    calculateAllStats(d_neg_data, neg_rows, cols, stats_neg);
}
