#include <core.cuh>
#include <iostream>
#include <utils.cuh>
#include <algorithms.cuh>

void trainNaiveBayesFinal(
    float *train_data, // Datos de entrenamiento completos (column-major)
    int train_rows,
    int cols,
    int target_col_index,
    ColumnStats *stats_pos, // OUTPUT: estadísticas clase positiva
    ColumnStats *stats_neg  // OUTPUT: estadísticas clase negativa
)
{
    std::cout << "\n=== ENTRENAMIENTO FINAL ===" << std::endl;
    std::cout << "Entrenando con " << train_rows << " muestras completas..." << std::endl;

    // ============================================================
    // 1. SEPARAR TRAIN EN POS/NEG
    // ============================================================
    int train_pos_rows, train_neg_rows;

    // Usar tu función filterByValue existente
    float *train_pos = filterByValue(train_data, train_rows, cols,
                                     target_col_index, 1.0f, train_pos_rows);

    float *train_neg = filterByValue(train_data, train_rows, cols,
                                     target_col_index, 0.0f, train_neg_rows);

    std::cout << "  Clase Positiva: " << train_pos_rows << " muestras" << std::endl;
    std::cout << "  Clase Negativa: " << train_neg_rows << " muestras" << std::endl;

    // ============================================================
    // 2. COPIAR A GPU
    // ============================================================
    float *d_train_pos, *d_train_neg;

    cudaMalloc(&d_train_pos, train_pos_rows * cols * sizeof(float));
    cudaMalloc(&d_train_neg, train_neg_rows * cols * sizeof(float));

    cudaMemcpy(d_train_pos, train_pos, train_pos_rows * cols * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_train_neg, train_neg, train_neg_rows * cols * sizeof(float),
               cudaMemcpyHostToDevice);

    // ============================================================
    // 3. CALCULAR ESTADÍSTICAS EN GPU
    // ============================================================
    // Usar tu función computeClassStatistics existente
    computeClassStatistics(d_train_pos, train_pos_rows,
                           d_train_neg, train_neg_rows,
                           cols, stats_pos, stats_neg);

    std::cout << "Estadísticas calculadas." << std::endl;

    // ============================================================
    // 4. LIBERAR MEMORIA
    // ============================================================
    delete[] train_pos;
    delete[] train_neg;
    cudaFree(d_train_pos);
    cudaFree(d_train_neg);

    std::cout << "=== ENTRENAMIENTO FINAL COMPLETADO ===" << std::endl;
}