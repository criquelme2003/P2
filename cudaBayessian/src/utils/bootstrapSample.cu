#include <vector>
#include <random>

// Versión alternativa más eficiente usando array de flags
void bootstrapSample(int n_samples,
                     std::vector<int> &bootstrap_indices,
                     std::vector<int> &oob_indices,
                     std::mt19937 &rng)
{

    bootstrap_indices.clear();
    oob_indices.clear();

    bootstrap_indices.reserve(n_samples);
    oob_indices.reserve(n_samples / 3);

    // ============================================================
    // 1. GENERAR ÍNDICES BOOTSTRAP CON REEMPLAZO
    // ============================================================
    std::uniform_int_distribution<int> dist(0, n_samples - 1);

    // Array de flags para marcar qué índices fueron seleccionados
    std::vector<bool> was_selected(n_samples, false);

    for (int i = 0; i < n_samples; i++)
    {
        int random_idx = dist(rng);
        bootstrap_indices.push_back(random_idx);
        was_selected[random_idx] = true; // Marcar como seleccionado
    }

    // ============================================================
    // 2. IDENTIFICAR MUESTRAS OOB
    // ============================================================
    // Recorrer array de flags en O(n)
    for (int i = 0; i < n_samples; i++)
    {
        if (!was_selected[i])
        {
            oob_indices.push_back(i);
        }
    }
}


void bootstrapSampleBalanced(
    int n_samples,
    const int* labels,              // Array con labels (0 o 1)
    std::vector<int>& bootstrap_indices,
    std::vector<int>& oob_indices,
    std::mt19937& rng
) {
    bootstrap_indices.clear();
    oob_indices.clear();
    
    // ============================================================
    // 1. SEPARAR ÍNDICES POR CLASE
    // ============================================================
    std::vector<int> pos_indices;
    std::vector<int> neg_indices;
    
    for (int i = 0; i < n_samples; i++) {
        if (labels[i] == 1) {
            pos_indices.push_back(i);
        } else {
            neg_indices.push_back(i);
        }
    }
    
    int n_pos = pos_indices.size();
    int n_neg = neg_indices.size();
    
    // ============================================================
    // 2. CALCULAR TAMAÑO BALANCEADO
    // ============================================================
    // Cada clase aportará la mitad del bootstrap
    int samples_per_class = n_samples / 2;
    
    bootstrap_indices.reserve(samples_per_class * 2);
    
    // ============================================================
    // 3. MUESTREAR 50% POSITIVOS
    // ============================================================
    std::uniform_int_distribution<int> dist_pos(0, n_pos - 1);
    std::vector<bool> pos_selected(n_pos, false);
    
    for (int i = 0; i < samples_per_class; i++) {
        int random_idx = dist_pos(rng);
        bootstrap_indices.push_back(pos_indices[random_idx]);
        pos_selected[random_idx] = true;
    }
    
    // ============================================================
    // 4. MUESTREAR 50% NEGATIVOS
    // ============================================================
    std::uniform_int_distribution<int> dist_neg(0, n_neg - 1);
    std::vector<bool> neg_selected(n_neg, false);
    
    for (int i = 0; i < samples_per_class; i++) {
        int random_idx = dist_neg(rng);
        bootstrap_indices.push_back(neg_indices[random_idx]);
        neg_selected[random_idx] = true;
    }
    
    // ============================================================
    // 5. IDENTIFICAR OOB
    // ============================================================
    // OOB positivos
    for (int i = 0; i < n_pos; i++) {
        if (!pos_selected[i]) {
            oob_indices.push_back(pos_indices[i]);
        }
    }
    
    // OOB negativos
    for (int i = 0; i < n_neg; i++) {
        if (!neg_selected[i]) {
            oob_indices.push_back(neg_indices[i]);
        }
    }
}

// void testBootstrapSample(){
//         std::cout << "\n=== TEST BOOTSTRAP SAMPLE ===" << std::endl;
    
//     int n_samples = 100;
//     std::mt19937 rng(123);
    
//     std::vector<int> bootstrap_indices;
//     std::vector<int> oob_indices;
    
//     // Ejecutar bootstrap
//     bootstrapSample(n_samples, bootstrap_indices, oob_indices, rng);
    
//     std::cout << "Total muestras: " << n_samples << std::endl;
//     std::cout << "Bootstrap size: " << bootstrap_indices.size() << std::endl;
//     std::cout << "OOB size: " << oob_indices.size() << std::endl;
//     std::cout << "OOB percentage: " << (float)oob_indices.size() / n_samples * 100 << "%" << std::endl;
    
//     // Verificar que no hay duplicados en OOB
//     std::set<int> oob_set(oob_indices.begin(), oob_indices.end());
//     std::cout << "OOB únicos: " << oob_set.size() << " (debe ser igual a OOB size)" << std::endl;
    
//     // Contar frecuencia de cada índice en bootstrap
//     std::vector<int> frequency(n_samples, 0);
//     for (int idx : bootstrap_indices) {
//         frequency[idx]++;
//     }
    
//     // Mostrar algunos ejemplos
//     std::cout << "\nEjemplos de frecuencias en bootstrap:" << std::endl;
//     for (int i = 0; i < 10; i++) {
//         std::cout << "  Índice " << i << ": aparece " << frequency[i] << " veces";
//         if (frequency[i] == 0) {
//             std::cout << " (OOB)";
//         }
//         std::cout << std::endl;
//     }
    
//     // Primeros 20 índices bootstrap
//     std::cout << "\nPrimeros 20 índices bootstrap: ";
//     for (int i = 0; i < 20 && i < bootstrap_indices.size(); i++) {
//         std::cout << bootstrap_indices[i] << " ";
//     }
//     std::cout << std::endl;
    
//     // Primeros 10 índices OOB
//     std::cout << "Primeros 10 índices OOB: ";
//     for (int i = 0; i < 10 && i < oob_indices.size(); i++) {
//         std::cout << oob_indices[i] << " ";
//     }
//     std::cout << std::endl;
    
//     std::cout << "=== FIN TEST ===" << std::endl;
// }