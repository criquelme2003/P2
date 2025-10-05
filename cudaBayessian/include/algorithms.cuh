#include <core.cuh>
#include <random>
ColumnStats calculateColumnStats(const float *d_data, int rows, int columnIndex);

void calculateAllStats(const float *d_data, int rows, int cols,
                       ColumnStats *stats_array);

void computeClassStatistics(const float *d_pos_data, int pos_rows,
                            const float *d_neg_data, int neg_rows,
                            int cols,
                            ColumnStats *stats_pos,
                            ColumnStats *stats_neg);

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
                         float epsilon = 0.0f);

void evaluateThresholdsGPU(
    const float *h_log_diffs,  // Host: diferencias log
    const int *h_y_true,       // Host: labels reales
    const float *h_thresholds, // Host: umbrales a probar
    float *h_youden_scores,    // Host: OUTPUT Youden scores
    int n_samples,
    int n_thresholds);

void predictLogDifferences(
    const float *h_test_data,       // Host: datos de test (row-major)
    const ColumnStats *h_stats_pos, // Host: estadísticas clase positiva
    const ColumnStats *h_stats_neg, // Host: estadísticas clase negativa
    float prior_pos,
    float prior_neg,
    float *h_log_diffs, // Host: OUTPUT diferencias log
    int n_samples,
    int cols,
    int target_col_index);

std::vector<MonteCarloIteration> monteCarloThresholdSearch(
    float *train_data, // Datos de entrenamiento (column-major)
    int train_rows,
    int cols,
    int target_col_index,
    int num_iterations = 100,
    unsigned int seed = 123);

void analyzeMonteCarloResults(const std::vector<MonteCarloIteration> &results);

void trainNaiveBayesFinal(
    float* train_data,           // Datos de entrenamiento completos (column-major)
    int train_rows,
    int cols,
    int target_col_index,
    ColumnStats* stats_pos,      // OUTPUT: estadísticas clase positiva
    ColumnStats* stats_neg       // OUTPUT: estadísticas clase negativa
);


Metrics calcularMetricas(const int *h_y_true, const int *h_y_pred, int n);

void printMetrics(const Metrics &m);
