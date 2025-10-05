#include <iostream>
#include <vector>
#include <core.cuh>
#include <random>
#include <fstream>
#include <utils.cuh>
#include <algorithms.cuh>

// Función principal: Monte Carlo para búsqueda de umbral óptimo
std::vector<MonteCarloIteration> monteCarloThresholdSearch(
    float *train_data, // Datos de entrenamiento (column-major)
    int train_rows,
    int cols,
    int target_col_index,
    int num_iterations,
    unsigned int seed)
{
    std::vector<MonteCarloIteration> results(num_iterations);
    std::mt19937 rng(seed);

    for (int iter = 0; iter < num_iterations; iter++)
    {
        std::cout
            << "Iteración Monte Carlo: " << iter + 1 << "/" << num_iterations << std::endl;

        // Inicializar resultado de esta iteración
        MonteCarloIteration &current_result = results[iter];

        // ============================================================
        // 1. BOOTSTRAP SAMPLING
        // ============================================================
        std::vector<int> bootstrap_indices;
        std::vector<int> oob_indices;
        bootstrapSample(train_rows, bootstrap_indices, oob_indices, rng);

        int n_bootstrap = bootstrap_indices.size(); // ≈ train_rows
        int n_oob = oob_indices.size();             // ≈ 0.37 * train_rows

        current_result.n_bootstrap = n_bootstrap;
        current_result.n_oob = n_oob;

        std::cout << "  Bootstrap: " << n_bootstrap << " muestras" << std::endl;
        std::cout << "  OOB: " << n_oob << " muestras" << std::endl;

        // ============================================================
        // 2. CREAR SUBSET DE BOOTSTRAP
        // ============================================================
        float *bootstrap_data = createSubset(train_data, train_rows, cols, bootstrap_indices);

        // ============================================================
        // 3. SEPARAR BOOTSTRAP EN POS/NEG
        // ============================================================
        int bootstrap_pos_rows, bootstrap_neg_rows;

        float *bootstrap_pos = filterByValue(bootstrap_data, n_bootstrap, cols,
                                             target_col_index, 1.0f, bootstrap_pos_rows);

        float *bootstrap_neg = filterByValue(bootstrap_data, n_bootstrap, cols,
                                             target_col_index, 0.0f, bootstrap_neg_rows);

        current_result.bootstrap_pos_count = bootstrap_pos_rows;
        current_result.bootstrap_neg_count = bootstrap_neg_rows;

        std::cout << "  Bootstrap Pos: " << bootstrap_pos_rows << std::endl;
        std::cout << "  Bootstrap Neg: " << bootstrap_neg_rows << std::endl;

        // ============================================================
        // 4. COPIAR DATOS BOOTSTRAP A GPU
        // ============================================================
        float *d_bootstrap_pos, *d_bootstrap_neg;
        cudaMalloc(&d_bootstrap_pos, bootstrap_pos_rows * cols * sizeof(float));
        cudaMalloc(&d_bootstrap_neg, bootstrap_neg_rows * cols * sizeof(float));
        cudaMemcpy(d_bootstrap_pos, bootstrap_pos, bootstrap_pos_rows * cols * sizeof(float),
                   cudaMemcpyHostToDevice);
        cudaMemcpy(d_bootstrap_neg, bootstrap_neg, bootstrap_neg_rows * cols * sizeof(float),
                   cudaMemcpyHostToDevice);

        // ============================================================
        // 5. CALCULAR ESTADÍSTICAS EN GPU
        // ============================================================
        ColumnStats *stats_pos = new ColumnStats[cols];
        ColumnStats *stats_neg = new ColumnStats[cols];

        computeClassStatistics(d_bootstrap_pos, bootstrap_pos_rows,
                               d_bootstrap_neg, bootstrap_neg_rows,
                               cols, stats_pos, stats_neg);

        // ============================================================
        // 6. EXTRAER DATOS OOB
        // ============================================================
        float *oob_data = createSubset(train_data, train_rows, cols, oob_indices);

        // Extraer labels OOB
        int *oob_labels = new int[n_oob];
        for (int i = 0; i < n_oob; i++)
        {
            int original_idx = oob_indices[i];
            // Column-major: target_col_index * train_rows + original_idx
            oob_labels[i] = (int)train_data[target_col_index * train_rows + original_idx];
        }

        // ============================================================
        // 7. TRANSPONER OOB A ROW-MAJOR
        // ============================================================
        float *oob_data_rowmajor = columnMajorToRowMajor(oob_data, n_oob, cols);

        // ============================================================
        // 8. PREDECIR LOG_DIFFS EN OOB (GPU)
        // ============================================================
        float *log_diffs = new float[n_oob];

        float prior_pos = (float)bootstrap_pos_rows / n_bootstrap;
        float prior_neg = (float)bootstrap_neg_rows / n_bootstrap;

        // Predecir diferencias logarítmicas
        predictLogDifferences(
            oob_data_rowmajor, // Datos OOB en row-major
            stats_pos,         // Estadísticas clase positiva
            stats_neg,         // Estadísticas clase negativa
            prior_pos,
            prior_neg,
            log_diffs, // OUTPUT: diferencias
            n_oob,     // Número de muestras OOB
            cols,
            target_col_index);

        // ============================================================
        // 9. GRID SEARCH DE UMBRALES (GPU)
        // ============================================================
        float threshold_min = -5.0f;
        float threshold_max = 5.0f;
        float threshold_step = 0.1f;
        int n_thresholds = (int)((threshold_max - threshold_min) / threshold_step) + 1;

        float *thresholds = new float[n_thresholds];
        for (int i = 0; i < n_thresholds; i++)
        {
            thresholds[i] = threshold_min + i * threshold_step;
        }

        float *youden_scores = new float[n_thresholds];

        evaluateThresholdsGPU(log_diffs, oob_labels, thresholds,
                              youden_scores, n_oob, n_thresholds);

        // ============================================================
        // 10. GUARDAR TODOS LOS UMBRALES Y YOUDEN SCORES
        // ============================================================
        current_result.all_thresholds.resize(n_thresholds);
        current_result.all_youden_scores.resize(n_thresholds);

        for (int i = 0; i < n_thresholds; i++)
        {
            current_result.all_thresholds[i] = thresholds[i];
            current_result.all_youden_scores[i] = youden_scores[i];
        }

        // ============================================================
        // 11. ENCONTRAR UMBRAL CON MÁXIMO YOUDEN
        // ============================================================
        float best_threshold = thresholds[0];
        float best_youden = youden_scores[0];

        for (int i = 1; i < n_thresholds; i++)
        {
            if (youden_scores[i] > best_youden)
            {
                best_youden = youden_scores[i];
                best_threshold = thresholds[i];
            }
        }

        current_result.optimal_threshold = best_threshold;
        current_result.best_youden = best_youden;

        std::cout << "  Umbral óptimo: " << best_threshold
                  << " (Youden: " << best_youden << ")" << std::endl;

        // ============================================================
        // 12. LIBERAR MEMORIA DE ESTA ITERACIÓN
        // ============================================================
        delete[] bootstrap_data;
        delete[] bootstrap_pos;
        delete[] bootstrap_neg;
        delete[] oob_data;
        delete[] oob_data_rowmajor;
        delete[] oob_labels;
        delete[] log_diffs;
        delete[] thresholds;
        delete[] youden_scores;
        delete[] stats_pos;
        delete[] stats_neg;
        cudaFree(d_bootstrap_pos);
        cudaFree(d_bootstrap_neg);

        std::cout << std::endl;
    }

    return results;
}

// Función para calcular estadísticas finales de los umbrales
void analyzeMonteCarloResults(const std::vector<MonteCarloIteration> &results)
{
    float mean_threshold = 0.0f;
    float mean_youden = 0.0f;

    for (const auto &result : results)
    {
        mean_threshold += result.optimal_threshold;
        mean_youden += result.best_youden;
    }

    mean_threshold /= results.size();
    mean_youden /= results.size();

    // Calcular desviación estándar
    float std_threshold = 0.0f;
    for (const auto &result : results)
    {
        float diff = result.optimal_threshold - mean_threshold;
        std_threshold += diff * diff;
    }
    std_threshold = std::sqrt(std_threshold / results.size());

    std::cout << "\n========================================" << std::endl;
    std::cout << "RESULTADOS MONTE CARLO" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Umbral óptimo final: " << mean_threshold
              << " ± " << std_threshold << std::endl;
    std::cout << "Youden promedio: " << mean_youden << std::endl;
    std::cout << "========================================\n"
              << std::endl;
}
