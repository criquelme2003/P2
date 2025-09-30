#include <utils.cuh>
#include <iostream>
#include <vector>

int main()
{

    int rows, cols;
    std::vector<std::string> headers;

    float *h_data = readCSV("diabetes.csv", rows, cols, headers);
    transposeInPlace(h_data, rows, cols);

    std::vector<int> train_indices;
    std::vector<int> test_indices;

    // Realizar split estratificado
    int targetColumnIndex = cols - 1;
    printf("targetCol: %d\n", targetColumnIndex);
    stratifiedSplit(h_data, rows, cols, targetColumnIndex,
                    train_indices, test_indices, 0.8f, 123);

    std::cout << "\nTamaños de conjuntos:" << std::endl;
    std::cout << "Entrenamiento: " << train_indices.size() << std::endl;
    std::cout << "Prueba: " << test_indices.size() << std::endl;

    // Crear subconjuntos
    float *datos_train = createSubset(h_data, rows, cols, train_indices);
    float *datos_test = createSubset(h_data, rows, cols, test_indices);

    // Verificar proporciones
    printClassProportions(datos_train, train_indices.size(),
                          targetColumnIndex, "entrenamiento");
    printClassProportions(datos_test, test_indices.size(),
                          targetColumnIndex, "prueba");

    int train_pos_rows, train_neg_rows;
    int train_rows = train_indices.size();
    // Separar positivos
    float *train_pos = filterByValue(datos_train, train_rows, cols,
                                     targetColumnIndex, 1.0f, train_pos_rows);

    // Separar negativos
    float *train_neg = filterByValue(datos_train, train_rows, cols,
                                     targetColumnIndex, 0.0f, train_neg_rows);

    std::cout << "Positivos: " << train_pos_rows << std::endl;
    std::cout << "Negativos: " << train_neg_rows << std::endl;

    // No olvides liberar memoria
    delete[] train_pos;
    delete[] train_neg;

    // Liberar memoria
    delete[] h_data;
    delete[] datos_train;
    delete[] datos_test;

    return 0;
}