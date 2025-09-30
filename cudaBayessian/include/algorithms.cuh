#include <core.cuh>

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

Metrics calcularMetricas(const int *h_y_true, const int *h_y_pred, int n);

void printMetrics(const Metrics &m);
