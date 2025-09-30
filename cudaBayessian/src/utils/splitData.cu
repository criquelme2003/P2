
#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <utils.cuh>

void stratifiedSplit(const float *data, int rows, int cols,
                     int targetColumnIndex,
                     std::vector<int> &train_indices,
                     std::vector<int> &test_indices,
                     float trainRatio,
                     unsigned int seed)
{
    // Limpiar vectores por si acaso
    train_indices.clear();
    test_indices.clear();

    // Configurar generador de números aleatorios
    std::mt19937 rng(seed);

    // Obtener índices de cada clase
    std::vector<int> pos_indices = getIndicesByValue(data, rows, cols, targetColumnIndex, 1.0f);
    std::vector<int> neg_indices = getIndicesByValue(data, rows, cols, targetColumnIndex, 0.0f);

    std::cout << "Total positivos: " << pos_indices.size() << std::endl;
    std::cout << "Total negativos: " << neg_indices.size() << std::endl;

    // Mezclar los índices
    std::shuffle(pos_indices.begin(), pos_indices.end(), rng);
    std::shuffle(neg_indices.begin(), neg_indices.end(), rng);

    // Calcular tamaños de entrenamiento
    int train_pos_size = static_cast<int>(std::round(trainRatio * pos_indices.size()));
    int train_neg_size = static_cast<int>(std::round(trainRatio * neg_indices.size()));

    // Dividir índices positivos
    std::vector<int> train_pos(pos_indices.begin(), pos_indices.begin() + train_pos_size);
    std::vector<int> test_pos(pos_indices.begin() + train_pos_size, pos_indices.end());

    // Dividir índices negativos
    std::vector<int> train_neg(neg_indices.begin(), neg_indices.begin() + train_neg_size);
    std::vector<int> test_neg(neg_indices.begin() + train_neg_size, neg_indices.end());

    // Combinar índices de entrenamiento
    train_indices.insert(train_indices.end(), train_pos.begin(), train_pos.end());
    train_indices.insert(train_indices.end(), train_neg.begin(), train_neg.end());

    // Combinar índices de prueba
    test_indices.insert(test_indices.end(), test_pos.begin(), test_pos.end());
    test_indices.insert(test_indices.end(), test_neg.begin(), test_neg.end());

    // Mezclar los conjuntos finales
    std::shuffle(train_indices.begin(), train_indices.end(), rng);
    std::shuffle(test_indices.begin(), test_indices.end(), rng);
}

// crear subset a partir de indices.
float *createSubset(const float *data, int rows, int cols,
                    const std::vector<int> &indices)
{
    int newRows = indices.size();
    float *subset = new float[newRows * cols];

    // Copiar datos en column-major
    for (int j = 0; j < cols; j++)
    {
        for (int i = 0; i < newRows; i++)
        {
            int originalRow = indices[i];
            subset[j * newRows + i] = data[j * rows + originalRow];
        }
    }

    return subset;
}

// Función para filtrar filas por valor en una columna específica (column-major)
float* filterByValue(const float* data, int rows, int cols, 
                     int columnIndex, float targetValue, int& newRows) {
    // Primero contar cuántas filas cumplen la condición
    std::vector<int> matching_indices;
    
    for (int i = 0; i < rows; i++) {
        if (data[columnIndex * rows + i] == targetValue) {
            matching_indices.push_back(i);
        }
    }
    
    newRows = matching_indices.size();
    
    if (newRows == 0) {
        return nullptr;
    }
    
    // Crear nuevo dataset con solo las filas que cumplen la condición
    float* filtered = new float[newRows * cols];
    
    for (int j = 0; j < cols; j++) {
        for (int i = 0; i < newRows; i++) {
            int originalRow = matching_indices[i];
            filtered[j * newRows + i] = data[j * rows + originalRow];
        }
    }
    
    return filtered;
}