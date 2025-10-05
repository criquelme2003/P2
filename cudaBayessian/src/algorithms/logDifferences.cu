#include <cuda_runtime.h>
#include <cmath>
#include <iostream>
#include <core.cuh>
#include <kernels.cuh>
// Función wrapper para llamar al kernel desde CPU
void predictLogDifferences(
    const float* h_test_data,      // Host: datos de test (row-major)
    const ColumnStats* h_stats_pos, // Host: estadísticas clase positiva
    const ColumnStats* h_stats_neg, // Host: estadísticas clase negativa
    float prior_pos,
    float prior_neg,
    float* h_log_diffs,             // Host: OUTPUT diferencias log
    int n_samples,
    int cols,
    int target_col_index
) {
    // Calcular log priors
    float log_prior_pos = logf(prior_pos);
    float log_prior_neg = logf(prior_neg);
    
    // ============================================================
    // 1. Asignar memoria en GPU
    // ============================================================
    float *d_test_data, *d_log_diffs;
    ColumnStats *d_stats_pos, *d_stats_neg;
    
    cudaMalloc(&d_test_data, n_samples * cols * sizeof(float));
    cudaMalloc(&d_stats_pos, cols * sizeof(ColumnStats));
    cudaMalloc(&d_stats_neg, cols * sizeof(ColumnStats));
    cudaMalloc(&d_log_diffs, n_samples * sizeof(float));
    
    // ============================================================
    // 2. Copiar datos a GPU
    // ============================================================
    cudaMemcpy(d_test_data, h_test_data, n_samples * cols * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_stats_pos, h_stats_pos, cols * sizeof(ColumnStats),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_stats_neg, h_stats_neg, cols * sizeof(ColumnStats),
               cudaMemcpyHostToDevice);
    
    // ============================================================
    // 3. Configurar y lanzar kernel
    // ============================================================
    int threadsPerBlock = 256;
    int blocksPerGrid = (n_samples + threadsPerBlock - 1) / threadsPerBlock;
    
    predictLogDifferencesKernel<<<blocksPerGrid, threadsPerBlock>>>(
        d_test_data,
        d_stats_pos,
        d_stats_neg,
        log_prior_pos,
        log_prior_neg,
        d_log_diffs,
        n_samples,
        cols,
        target_col_index
    );
    
    // Verificar errores
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Error en kernel predictLogDifferencesKernel: "
                  << cudaGetErrorString(err) << std::endl;
    }
    
    cudaDeviceSynchronize();
    
    // ============================================================
    // 4. Copiar resultados de vuelta a CPU
    // ============================================================
    cudaMemcpy(h_log_diffs, d_log_diffs, n_samples * sizeof(float),
               cudaMemcpyDeviceToHost);
    
    // ============================================================
    // 5. Liberar memoria GPU
    // ============================================================
    cudaFree(d_test_data);
    cudaFree(d_stats_pos);
    cudaFree(d_stats_neg);
    cudaFree(d_log_diffs);
}

// Función de prueba
// void testPredictLogDifferences() {
//     std::cout << "\n=== TEST PREDICT LOG DIFFERENCES ===" << std::endl;
    
//     int n_samples = 10;
//     int cols = 3;  // 2 features + 1 target
//     int target_col = 2;
    
//     // Crear datos de prueba (row-major)
//     float* test_data = new float[n_samples * cols];
    
//     // Llenar con datos sintéticos
//     // Feature 0: valores bajos para clase 0, altos para clase 1
//     // Feature 1: valores bajos para clase 0, altos para clase 1
//     for (int i = 0; i < n_samples; i++) {
//         if (i < n_samples / 2) {
//             // Clase 0
//             test_data[i * cols + 0] = 10.0f + i;  // Feature 0
//             test_data[i * cols + 1] = 20.0f + i;  // Feature 1
//             test_data[i * cols + 2] = 0.0f;       // Target
//         } else {
//             // Clase 1
//             test_data[i * cols + 0] = 50.0f + i;  // Feature 0
//             test_data[i * cols + 1] = 60.0f + i;  // Feature 1
//             test_data[i * cols + 2] = 1.0f;       // Target
//         }
//     }
    
//     // Crear estadísticas sintéticas
//     ColumnStats* stats_pos = new ColumnStats[cols];
//     ColumnStats* stats_neg = new ColumnStats[cols];
    
//     // Clase negativa (valores bajos)
//     stats_neg[0].mean = 12.0f;
//     stats_neg[0].sd = 5.0f;
//     stats_neg[1].mean = 22.0f;
//     stats_neg[1].sd = 5.0f;
//     stats_neg[2].mean = 0.0f;  // Target (no se usa)
//     stats_neg[2].sd = 0.0f;
    
//     // Clase positiva (valores altos)
//     stats_pos[0].mean = 52.0f;
//     stats_pos[0].sd = 5.0f;
//     stats_pos[1].mean = 62.0f;
//     stats_pos[1].sd = 5.0f;
//     stats_pos[2].mean = 1.0f;  // Target (no se usa)
//     stats_pos[2].sd = 0.0f;
    
//     // Priors iguales
//     float prior_pos = 0.5f;
//     float prior_neg = 0.5f;
    
//     // Array para resultados
//     float* log_diffs = new float[n_samples];
    
//     // Ejecutar predicción
//     predictLogDifferences(test_data, stats_pos, stats_neg,
//                          prior_pos, prior_neg,
//                          log_diffs, n_samples, cols, target_col);
    
//     // Mostrar resultados
//     std::cout << "\nResultados:" << std::endl;
//     std::cout << "Sample\tFeature0\tFeature1\tTrue Class\tLog Diff\tPrediction" << std::endl;
//     std::cout << "-------------------------------------------------------------------------" << std::endl;
    
//     for (int i = 0; i < n_samples; i++) {
//         int true_class = (int)test_data[i * cols + target_col];
//         int prediction = (log_diffs[i] > 0.0f) ? 1 : 0;
        
//         std::cout << i << "\t"
//                   << test_data[i * cols + 0] << "\t\t"
//                   << test_data[i * cols + 1] << "\t\t"
//                   << true_class << "\t\t"
//                   << log_diffs[i] << "\t"
//                   << prediction
//                   << (prediction == true_class ? " ✓" : " ✗")
//                   << std::endl;
//     }
    
//     // Calcular accuracy
//     int correct = 0;
//     for (int i = 0; i < n_samples; i++) {
//         int true_class = (int)test_data[i * cols + target_col];
//         int prediction = (log_diffs[i] > 0.0f) ? 1 : 0;
//         if (prediction == true_class) correct++;
//     }
    
//     std::cout << "\nAccuracy con umbral 0.0: " << (float)correct / n_samples * 100 << "%" << std::endl;
    
//     // Interpretación
//     std::cout << "\nInterpretación:" << std::endl;
//     std::cout << "  log_diff > 0 → Clase POSITIVA más probable" << std::endl;
//     std::cout << "  log_diff < 0 → Clase NEGATIVA más probable" << std::endl;
//     std::cout << "  |log_diff| grande → Mayor confianza en la predicción" << std::endl;
    
//     // Liberar memoria
//     delete[] test_data;
//     delete[] stats_pos;
//     delete[] stats_neg;
//     delete[] log_diffs;
    
//     std::cout << "=== FIN TEST ===" << std::endl;
// }