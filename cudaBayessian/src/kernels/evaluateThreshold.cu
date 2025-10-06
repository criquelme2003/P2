__global__ void evaluateThresholdsKernel(
    const float* log_diffs,      // [n_samples] diferencias log(P_pos) - log(P_neg)
    const int* y_true,            // [n_samples] labels reales (0 o 1)
    const float* thresholds,      // [n_thresholds] umbrales a probar
    float* youden_scores,         // [n_thresholds] OUTPUT: Youden scores
    int n_samples,
    int n_thresholds
) {
    // Cada thread evalúa UN umbral
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid >= n_thresholds) return;
    
    float threshold = thresholds[tid];
    
    // ============================================================
    // Calcular Confusion Matrix para este umbral
    // ============================================================
    int TP = 0, TN = 0, FP = 0, FN = 0;
    
    for (int i = 0; i < n_samples; i++) {
        // Clasificar usando este umbral
        int prediction = (log_diffs[i] > threshold) ? 1 : 0;
        int truth = y_true[i];
        
        // Actualizar confusion matrix
        if (prediction == 1 && truth == 1) {
            TP++;
        } else if (prediction == 0 && truth == 0) {
            TN++;
        } else if (prediction == 1 && truth == 0) {
            FP++;
        } else { // prediction == 0 && truth == 1
            FN++;
        }
    }
    
    // ============================================================
    // Calcular Youden Index
    // ============================================================
    // Youden = Sensitivity + Specificity - 1
    // Sensitivity (Recall) = TP / (TP + FN)
    // Specificity = TN / (TN + FP)
    
    float sensitivity = 0.0f;
    float specificity = 0.0f;
    
    if ((TP + FN) > 0) {
        sensitivity = (float)TP / (float)(TP + FN);
    }
    
    if ((TN + FP) > 0) {
        specificity = (float)TN / (float)(TN + FP);
    }
    
    float youden = sensitivity + specificity - 1.0f;
    
    // Guardar resultado
    youden_scores[tid] = youden;
}

__global__ void evaluateThresholdsF2Kernel(
    const float* log_diffs,
    const int* y_true,
    const float* thresholds,
    float* f2_scores,           // CAMBIO: Antes era youden_scores
    int n_samples,
    int n_thresholds
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid >= n_thresholds) return;
    
    float threshold = thresholds[tid];
    
    // Calcular Confusion Matrix
    int TP = 0, TN = 0, FP = 0, FN = 0;
    
    for (int i = 0; i < n_samples; i++) {
        int prediction = (log_diffs[i] > threshold) ? 1 : 0;
        int truth = y_true[i];
        
        if (prediction == 1 && truth == 1) TP++;
        else if (prediction == 0 && truth == 0) TN++;
        else if (prediction == 1 && truth == 0) FP++;
        else FN++;
    }
    
    // ============================================================
    // CAMBIO: Calcular F2-Score en lugar de Youden
    // ============================================================
    // F2 = (1 + 2²) × (precision × recall) / (2² × precision + recall)
    // F2 = 5 × (precision × recall) / (4 × precision + recall)
    
    float precision = 0.0f;
    float recall = 0.0f;
    
    if ((TP + FP) > 0) {
        precision = (float)TP / (float)(TP + FP);
    }
    
    if ((TP + FN) > 0) {
        recall = (float)TP / (float)(TP + FN);
    }
    
    float f2 = 0.0f;
    if (precision + recall > 0.0f) {
        f2 = 5.0f * precision * recall / (4.0f * precision + recall);
    }
    
    f2_scores[tid] = f2;
}