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