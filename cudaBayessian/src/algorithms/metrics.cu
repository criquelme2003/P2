#include <core.cuh>
#include <kernels.cuh>
#include <iostream>
// Función para calcular métricas en GPU
Metrics calcularMetricas(const int* h_y_true, const int* h_y_pred, int n) {
    Metrics metrics;
    
    // Copiar datos a GPU
    int *d_y_true, *d_y_pred;
    cudaMalloc(&d_y_true, n * sizeof(int));
    cudaMalloc(&d_y_pred, n * sizeof(int));
    cudaMemcpy(d_y_true, h_y_true, n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y_pred, h_y_pred, n * sizeof(int), cudaMemcpyHostToDevice);
    
    // Crear array para contadores [TP, TN, FP, FN]
    int *d_counts;
    cudaMalloc(&d_counts, 4 * sizeof(int));
    cudaMemset(d_counts, 0, 4 * sizeof(int));
    
    // Configurar kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    int sharedMemSize = 4 * sizeof(int);
    
    // Ejecutar kernel
    countConfusionMatrixKernel<<<blocksPerGrid, threadsPerBlock, sharedMemSize>>>(
        d_y_true, d_y_pred, d_counts, d_counts + 1, d_counts + 2, d_counts + 3, n
    );
    
    // Copiar resultados
    int h_counts[4];
    cudaMemcpy(h_counts, d_counts, 4 * sizeof(int), cudaMemcpyDeviceToHost);
    
    metrics.TP = h_counts[0];
    metrics.TN = h_counts[1];
    metrics.FP = h_counts[2];
    metrics.FN = h_counts[3];
    
    // Calcular métricas en CPU (cálculos simples, no vale la pena usar GPU)
    float tp = (float)metrics.TP;
    float tn = (float)metrics.TN;
    float fp = (float)metrics.FP;
    float fn = (float)metrics.FN;
    
    metrics.precision = (tp + fp == 0) ? 0.0f : tp / (tp + fp);
    metrics.recall = (tp + fn == 0) ? 0.0f : tp / (tp + fn);
    metrics.f1_score = (metrics.precision + metrics.recall == 0) ? 0.0f : 
                       2.0f * metrics.precision * metrics.recall / (metrics.precision + metrics.recall);
    metrics.accuracy = (tp + tn + fp + fn == 0) ? 0.0f : (tp + tn) / (tp + tn + fp + fn);
    
    // Liberar memoria
    cudaFree(d_y_true);
    cudaFree(d_y_pred);
    cudaFree(d_counts);
    
    return metrics;
}

// Función para imprimir métricas
void printMetrics(const Metrics& m) {
    std::cout << "\n=== Matriz de Confusión ===" << std::endl;
    std::cout << "TP: " << m.TP << " | FP: " << m.FP << std::endl;
    std::cout << "FN: " << m.FN << " | TN: " << m.TN << std::endl;
    
    std::cout << "\n=== Métricas ===" << std::endl;
    std::cout << "Accuracy:  " << m.accuracy << std::endl;
    std::cout << "Precision: " << m.precision << std::endl;
    std::cout << "Recall:    " << m.recall << std::endl;
    std::cout << "F1-Score:  " << m.f1_score << std::endl;
}