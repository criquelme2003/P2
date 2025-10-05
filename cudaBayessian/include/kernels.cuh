
#include <core.cuh>

__global__ void columnSumSquaresKernel(const float *data, int rows,
                                       int col_offset, float mean,
                                       float *result);

__global__ void columnSumKernel(const float *data, int rows, int col_offset,
                                float *result);

__global__ void naiveBayesKernel(const float *test_data,
                                 const ColumnStats *stats_pos,
                                 const ColumnStats *stats_neg,
                                 float log_prior_pos,
                                 float log_prior_neg,
                                 int *predictions,
                                 float *log_likelihood_pos,
                                 float *log_likelihood_neg,
                                 int n_test,
                                 int cols,
                                 int target_col_index,
                                 float epsilon);

__global__ void countConfusionMatrixKernel(const int *y_true,
                                           const int *y_pred,
                                           int *TP, int *TN,
                                           int *FP, int *FN,
                                           int n);

__global__ void evaluateThresholdsKernel(
    const float *log_diffs,  // [n_samples] diferencias log(P_pos) - log(P_neg)
    const int *y_true,       // [n_samples] labels reales (0 o 1)
    const float *thresholds, // [n_thresholds] umbrales a probar
    float *youden_scores,    // [n_thresholds] OUTPUT: Youden scores
    int n_samples,
    int n_thresholds);

__global__ void predictLogDifferencesKernel(
    const float *test_data,       // [n_samples × cols] datos en row-major
    const ColumnStats *stats_pos, // [cols] estadísticas clase positiva
    const ColumnStats *stats_neg, // [cols] estadísticas clase negativa
    float log_prior_pos,
    float log_prior_neg,
    float *log_diffs, // [n_samples] OUTPUT: diferencias log
    int n_samples,
    int cols,
    int target_col_index);