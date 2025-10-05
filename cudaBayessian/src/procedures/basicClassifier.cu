#include <utils.cuh>
#include <core.cuh>
#include <algorithms.cuh>
#include <core.cuh>
#include <iostream>
#include <vector>
int basic()
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

    float prior_pos = (float)train_pos_rows / train_rows;
    float prior_neg = (float)train_neg_rows / train_rows;

    // trasponer data a orignial para entrenamiento
    int n_test = test_indices.size();
    float *datos_test_rowmajor = columnMajorToRowMajor(datos_test, n_test, cols);

    int *predictions = new int[n_test];
    float *log_likelihood_pos = new float[n_test];
    float *log_likelihood_neg = new float[n_test];

    // Ejecutar clasificación
    clasificarBayesiano(datos_test_rowmajor, stats_pos, stats_neg,
                        prior_pos, prior_neg,
                        predictions, log_likelihood_pos, log_likelihood_neg,
                        n_test, cols, targetColumnIndex, -0.8f);

    int *y_true = new int[n_test];
    for (int i = 0; i < n_test; i++)
    {
        y_true[i] = (int)datos_test_rowmajor[i * cols + targetColumnIndex];
    }

    // Calcular métricas
    Metrics metricas = calcularMetricas(y_true, predictions, n_test);

    // Mostrar resultados
    printMetrics(metricas);

    printf("datos totales %d\n", n_test);
    // Liberar
    delete[] y_true;

    // Liberar memoria
    delete[] predictions;
    delete[] log_likelihood_pos;
    delete[] log_likelihood_neg;

    // Liberar memoria
    delete[] stats_pos;
    delete[] stats_neg;
    cudaFree(d_pos_data);
    cudaFree(d_neg_data);
    delete[] train_pos;
    delete[] train_neg;
    delete[] h_data;
    delete[] datos_train;
    delete[] datos_test;

    return 0;
}


// Función para entrenar Naive Bayes con TODO el conjunto de entrenamiento
