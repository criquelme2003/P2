
#pragma once

struct ColumnStats
{
    float mean;
    float sd;
};

struct Metrics
{
    int TP, TN, FP, FN;
    float precision, recall, f1_score, accuracy;
};

struct MonteCarloIteration
{
    // Mejor umbral y su Youden para esta iteración
    float optimal_threshold;
    float best_youden;

    // TODOS los umbrales probados y sus Youden scores
    std::vector<float> all_thresholds;
    std::vector<float> all_youden_scores;

    // Información adicional de la iteración
    int n_bootstrap;
    int n_oob;
    int bootstrap_pos_count;
    int bootstrap_neg_count;
};
