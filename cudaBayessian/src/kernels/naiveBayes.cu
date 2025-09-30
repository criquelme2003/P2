#include <core.cuh>
#include <cmath>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

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
                                 float epsilon)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= n_test)
        return;

    // Inicializar con los priors
    float log_prob_pos = log_prior_pos;
    float log_prob_neg = log_prior_neg;

    // Iterar sobre todas las columnas EXCEPTO la columna target
    for (int j = 0; j < cols; j++)
    {
        // Saltar la columna target
        if (j == target_col_index)
            continue;

        // Acceso row-major: test_data[i * cols + j]
        float valor = test_data[i * cols + j];

        // Calcular probabilidad para clase positiva usando distribución normal
        float mean_pos = stats_pos[j].mean;
        float sd_pos = stats_pos[j].sd;

        // PDF de distribución normal
        float z_pos = (valor - mean_pos) / sd_pos;
        float prob_pos = (1.0f / (sd_pos * sqrtf(2.0f * M_PI))) * expf(-0.5f * z_pos * z_pos);

        // Evitar log(0)
        prob_pos = fmaxf(prob_pos, 1e-30f);
        log_prob_pos += logf(prob_pos);

        // Calcular probabilidad para clase negativa
        float mean_neg = stats_neg[j].mean;
        float sd_neg = stats_neg[j].sd;

        float z_neg = (valor - mean_neg) / sd_neg;
        float prob_neg = (1.0f / (sd_neg * sqrtf(2.0f * M_PI))) * expf(-0.5f * z_neg * z_neg);

        prob_neg = fmaxf(prob_neg, 1e-30f);
        log_prob_neg += logf(prob_neg);
    }

    // Guardar log-verosimilitudes
    log_likelihood_pos[i] = log_prob_pos;
    log_likelihood_neg[i] = log_prob_neg;

    // Clasificar según la diferencia y el umbral epsilon
    float diferencia = log_prob_pos - log_prob_neg;
    predictions[i] = (diferencia > epsilon) ? 1 : 0; // 1 = pos, 0 = neg
}
