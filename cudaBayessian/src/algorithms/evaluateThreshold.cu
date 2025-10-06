#include <cuda_runtime.h>
#include <iostream>
#include <random>
#include <kernels.cuh>
// Kernel para evaluar todos los umbrales en paralelo

// Función wrapper para llamar al kernel desde CPU
void evaluateThresholdsGPU(
    const float *h_log_diffs,  // Host: diferencias log
    const int *h_y_true,       // Host: labels reales
    const float *h_thresholds, // Host: umbrales a probar
    float *h_scores,           // Host: OUTPUT Youden scores
    int n_samples,
    int n_thresholds)
{
    // ============================================================
    // 1. Asignar memoria en GPU
    // ============================================================
    float *d_log_diffs, *d_thresholds, *d_scores;
    int *d_y_true;

    cudaMalloc(&d_log_diffs, n_samples * sizeof(float));
    cudaMalloc(&d_y_true, n_samples * sizeof(int));
    cudaMalloc(&d_thresholds, n_thresholds * sizeof(float));
    cudaMalloc(&d_scores, n_thresholds * sizeof(float));

    // ============================================================
    // 2. Copiar datos a GPU
    // ============================================================
    cudaMemcpy(d_log_diffs, h_log_diffs, n_samples * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_y_true, h_y_true, n_samples * sizeof(int),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_thresholds, h_thresholds, n_thresholds * sizeof(float),
               cudaMemcpyHostToDevice);

    // ============================================================
    // 3. Configurar y lanzar kernel
    // ============================================================
    // 1 thread por umbral
    int threadsPerBlock = 256;
    int blocksPerGrid = (n_thresholds + threadsPerBlock - 1) / threadsPerBlock;

    evaluateThresholdsF2Kernel<<<blocksPerGrid, threadsPerBlock>>>(
        d_log_diffs,
        d_y_true,
        d_thresholds,
        d_scores, // CAMBIO
        n_samples,
        n_thresholds);

    // Verificar errores
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        std::cerr << "Error en kernel evaluateThresholdsKernel: "
                  << cudaGetErrorString(err) << std::endl;
    }

    cudaDeviceSynchronize();

    // ============================================================
    // 4. Copiar resultados de vuelta a CPU
    // ============================================================
    cudaMemcpy(h_scores, d_scores, n_thresholds * sizeof(float),
               cudaMemcpyDeviceToHost);

    // ============================================================
    // 5. Liberar memoria GPU
    // ============================================================
    cudaFree(d_log_diffs);
    cudaFree(d_y_true);
    cudaFree(d_thresholds);
    cudaFree(d_scores);
}

// Función de prueba
// void testEvaluateThresholdsGPU() {
//     std::cout << "\n=== TEST EVALUATE THRESHOLDS GPU ===" << std::endl;

//     // Crear datos de prueba
//     int n_samples = 100;
//     int n_thresholds = 11;  // De -1.0 a 1.0 en pasos de 0.2

//     float* log_diffs = new float[n_samples];
//     int* y_true = new int[n_samples];
//     float* thresholds = new float[n_thresholds];
//     float* youden_scores = new float[n_thresholds];

//     // Generar datos sintéticos
//     // Clase 0: log_diffs negativos (media -0.5)
//     // Clase 1: log_diffs positivos (media 0.5)
//     std::mt19937 rng(42);
//     std::normal_distribution<float> dist_neg(-0.5f, 0.3f);
//     std::normal_distribution<float> dist_pos(0.5f, 0.3f);

//     for (int i = 0; i < n_samples; i++) {
//         if (i < n_samples / 2) {
//             y_true[i] = 0;
//             log_diffs[i] = dist_neg(rng);
//         } else {
//             y_true[i] = 1;
//             log_diffs[i] = dist_pos(rng);
//         }
//     }

//     // Generar umbrales de -1.0 a 1.0
//     for (int i = 0; i < n_thresholds; i++) {
//         thresholds[i] = -1.0f + i * 0.2f;
//     }

//     // Ejecutar evaluación en GPU
//     evaluateThresholdsGPU(log_diffs, y_true, thresholds, youden_scores,
//                          n_samples, n_thresholds);

//     // Mostrar resultados
//     std::cout << "\nResultados:" << std::endl;
//     std::cout << "Threshold\tYouden Index" << std::endl;
//     std::cout << "--------------------------------" << std::endl;

//     float best_youden = youden_scores[0];
//     float best_threshold = thresholds[0];

//     for (int i = 0; i < n_thresholds; i++) {
//         std::cout << thresholds[i] << "\t\t" << youden_scores[i] << std::endl;

//         if (youden_scores[i] > best_youden) {
//             best_youden = youden_scores[i];
//             best_threshold = thresholds[i];
//         }
//     }

//     std::cout << "\nMejor umbral: " << best_threshold << std::endl;
//     std::cout << "Mejor Youden: " << best_youden << std::endl;

//     // Liberar memoria
//     delete[] log_diffs;
//     delete[] y_true;
//     delete[] thresholds;
//     delete[] youden_scores;

//     std::cout << "=== FIN TEST ===" << std::endl;
// }