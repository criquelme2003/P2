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