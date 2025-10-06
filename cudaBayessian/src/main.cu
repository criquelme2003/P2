#include <utils.cuh>
#include <core.cuh>
#include <algorithms.cuh>
#include <core.cuh>
#include <iostream>
#include <vector>
#include <set>
#include <random>
#include <chrono>
int main()
{

    int rows, cols;
    std::vector<std::string> headers;

    float *h_data = readCSV("diabetes.csv", rows, cols, headers);
    transposeInPlace(h_data, rows, cols);

    std::vector<int> train_indices;
    std::vector<int> test_indices;

    // Realizar split estratificado
    int targetColumnIndex = cols - 1;

    std::cout << "target col:" << targetColumnIndex << " | " << headers[targetColumnIndex] << std::endl;
    stratifiedSplit(h_data, rows, cols, targetColumnIndex,
                    train_indices, test_indices, 0.8f, 123);

    std::cout << "\nTamaños de conjuntos:" << std::endl;
    std::cout << "Entrenamiento: " << train_indices.size() << std::endl;
    std::cout << "Prueba: " << test_indices.size() << std::endl;

    // Crear subconjuntos
    float *datos_train = createSubset(h_data, rows, cols, train_indices);
    float *datos_test = createSubset(h_data, rows, cols, test_indices);

    // Verificar proporciones
    printClassProportions(datos_train, train_indices.size(),
                          targetColumnIndex, "entrenamiento");
    printClassProportions(datos_test, test_indices.size(),
                          targetColumnIndex, "prueba");

    int train_pos_rows, train_neg_rows;
    int train_rows = train_indices.size();
    // Separar positivos
    float *train_pos = filterByValue(datos_train, train_rows, cols,
                                     targetColumnIndex, 1.0f, train_pos_rows);

    // Separar negativos
    float *train_neg = filterByValue(datos_train, train_rows, cols, targetColumnIndex, 0.0f, train_neg_rows);

    std::cout << "Positivos: " << train_pos_rows << std::endl;
    std::cout << "Negativos: " << train_neg_rows << std::endl;

    float *d_pos_data, *d_neg_data;
    cudaMalloc(&d_pos_data, train_pos_rows * cols * sizeof(float));
    cudaMalloc(&d_neg_data, train_neg_rows * cols * sizeof(float));
    cudaMemcpy(d_pos_data, train_pos, train_pos_rows * cols * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_neg_data, train_neg, train_neg_rows * cols * sizeof(float),
               cudaMemcpyHostToDevice);

    // Arrays para guardar estadísticas
    ColumnStats *stats_pos = new ColumnStats[cols];
    ColumnStats *stats_neg = new ColumnStats[cols];

    // Calcular estadísticas
    computeClassStatistics(d_pos_data, train_pos_rows, d_neg_data, train_neg_rows, cols, stats_pos, stats_neg);

    // Mostrar resultados

    std::cout << "\nEstadísticas por clase:\n"
              << std::endl;
    for (int j = 0; j < cols; j++)
    {
        std::cout << "Variable: " << headers[j] << std::endl;
        std::cout << "  Clase Positiva - Media: " << stats_pos[j].mean
                  << ", SD: " << stats_pos[j].sd << std::endl;
        std::cout << "  Clase Negativa - Media: " << stats_neg[j].mean
                  << ", SD: " << stats_neg[j].sd << std::endl;
    }

    // Record the starting time point
    auto start = std::chrono::high_resolution_clock::now();

    std::vector<MonteCarloIteration> mc_results = monteCarloThresholdSearch(datos_train, train_rows, cols, targetColumnIndex, 1000);

    // Record the ending time point
    auto end = std::chrono::high_resolution_clock::now();

    // Calculate the duration
    auto duration = end - start;

    // Convert the duration to milliseconds and get the count
    auto milliseconds_elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(duration).count();

    std::cout << 1000 << " iterations performed in " << milliseconds_elapsed << "ms (global)" << std::endl;

    analyzeMonteCarloResults(mc_results);

    // Obtener umbral óptimo
    float optimal_threshold = 0.0f;
    for (const auto &r : mc_results)
    {
        optimal_threshold += r.optimal_threshold;
    }
    optimal_threshold /= mc_results.size();

    std::cout << "Usando umbral final (según F2-SCORE): " << optimal_threshold << std::endl;

    // ========================================
    // FASE 2: ENTRENAMIENTO FINAL
    // ========================================
    std::cout << "\n##########################################" << std::endl;
    std::cout << "# FASE 2: ENTRENAMIENTO FINAL           #" << std::endl;
    std::cout << "##########################################\n"
              << std::endl;

    ColumnStats *final_stats_pos = new ColumnStats[cols];
    ColumnStats *final_stats_neg = new ColumnStats[cols];

    // Entrenar con TODO el train set
    trainNaiveBayesFinal(datos_train, train_rows, cols, targetColumnIndex,
                         final_stats_pos, final_stats_neg);

    // Mostrar estadísticas finales
    std::cout << "\nEstadísticas del modelo final:" << std::endl;
    for (int j = 0; j < cols; j++)
    {
        std::cout << "Variable: " << headers[j] << std::endl;
        std::cout << "  Clase Positiva - Media: " << final_stats_pos[j].mean
                  << ", SD: " << final_stats_pos[j].sd << std::endl;
        std::cout << "  Clase Negativa - Media: " << final_stats_neg[j].mean
                  << ", SD: " << final_stats_neg[j].sd << std::endl;
    }

    // ========================================
    // FASE 3: EVALUACIÓN EN TEST
    // ========================================
    std::cout << "\n##########################################" << std::endl;
    std::cout << "# FASE 3: EVALUACIÓN EN TEST            #" << std::endl;
    std::cout << "##########################################\n"
              << std::endl;

    // Convertir test a row-major
    float *test_data_rowmajor = columnMajorToRowMajor(datos_test, test_indices.size(), cols);

    // Calcular priors finales
    int final_pos_count, final_neg_count;
    float *temp_pos = filterByValue(datos_train, train_rows, cols,
                                    targetColumnIndex, 1.0f, final_pos_count);
    float *temp_neg = filterByValue(datos_train, train_rows, cols,
                                    targetColumnIndex, 0.0f, final_neg_count);
    delete[] temp_pos;
    delete[] temp_neg;

    // float prior_pos = (float)final_pos_count / train_rows;
    // float prior_neg = (float)final_neg_count / train_rows;

    // priors balanceados
    float prior_pos = 0.5f;
    float prior_neg = 0.5f;

    std::cout << "Priors: P(pos)=" << prior_pos << ", P(neg)=" << prior_neg << std::endl;

    int test_rows = test_indices.size();
    // Arrays para predicciones
    int *predictions = new int[test_rows];
    float *log_likelihood_pos = new float[test_rows];
    float *log_likelihood_neg = new float[test_rows];

    // Clasificar usando umbral óptimo
    clasificarBayesiano(test_data_rowmajor,
                        final_stats_pos,
                        final_stats_neg,
                        prior_pos,
                        prior_neg,
                        predictions,
                        log_likelihood_pos,
                        log_likelihood_neg,
                        test_rows,
                        cols,
                        targetColumnIndex,
                        optimal_threshold); // ← Umbral de Monte Carlo

    // Extraer labels verdaderos
    int *y_true = new int[test_rows];
    for (int i = 0; i < test_rows; i++)
    {
        y_true[i] = (int)test_data_rowmajor[i * cols + targetColumnIndex];
    }

    // Calcular métricas
    Metrics metricas_optimized = calcularMetricas(y_true, predictions, test_rows);

    std::cout << "\n=== RESULTADOS CON UMBRAL OPTIMIZADO ===" << std::endl;
    std::cout << "Umbral usado: " << optimal_threshold << std::endl;
    printMetrics(metricas_optimized);

    delete[] final_stats_pos;
    delete[] final_stats_neg;
    delete[] test_data_rowmajor;
    delete[] predictions;
    delete[] log_likelihood_pos;
    delete[] log_likelihood_neg;
    delete[] y_true;
    delete[] datos_train;
    delete[] datos_test;
}