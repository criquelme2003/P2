
#pragma once

struct ColumnStats {
    float mean;
    float sd;
};

struct Metrics {
    int TP, TN, FP, FN;
    float precision, recall, f1_score, accuracy;
};
