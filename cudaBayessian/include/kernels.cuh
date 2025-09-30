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