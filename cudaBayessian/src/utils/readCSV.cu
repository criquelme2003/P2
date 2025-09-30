#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>

// Función para leer CSV con encabezados
float *readCSV(const std::string &filename, int &rows, int &cols,
               std::vector<std::string> &headers, char delimiter,
               bool hasHeader)
{
    std::ifstream file(filename);
    if (!file.is_open())
    {
        std::cerr << "Error: No se pudo abrir el archivo " << filename << std::endl;
        return nullptr;
    }

    std::vector<std::vector<float>> tempData;
    std::string line;

    // Leer encabezados si existen
    if (hasHeader && std::getline(file, line))
    {
        std::stringstream ss(line);
        std::string columnName;

        while (std::getline(ss, columnName, delimiter))
        {
            // Eliminar espacios al inicio y final
            columnName.erase(0, columnName.find_first_not_of(" \t\r\n"));
            columnName.erase(columnName.find_last_not_of(" \t\r\n") + 1);
            headers.push_back(columnName);
        }
    }

    // Leer datos
    while (std::getline(file, line))
    {
        std::vector<float> row;
        std::stringstream ss(line);
        std::string value;

        while (std::getline(ss, value, delimiter))
        {
            try
            {
                row.push_back(std::stof(value));
            }
            catch (const std::exception &e)
            {
                std::cerr << "Error al convertir valor: " << value << std::endl;
                file.close();
                return nullptr;
            }
        }

        if (!row.empty())
        {
            tempData.push_back(row);
        }
    }

    file.close();

    if (tempData.empty())
    {
        std::cerr << "Error: El archivo está vacío" << std::endl;
        return nullptr;
    }

    // Configurar dimensiones
    rows = tempData.size();
    cols = tempData[0].size();
    int totalSize = rows * cols;

    // Verificar que headers coincide con columnas
    if (hasHeader && !headers.empty() && headers.size() != cols)
    {
        std::cerr << "Advertencia: Número de encabezados (" << headers.size()
                  << ") no coincide con columnas (" << cols << ")" << std::endl;
    }

    // Asignar memoria y copiar datos
    float *h_data = new float[totalSize];

    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            h_data[i * cols + j] = tempData[i][j];
        }
    }

    return h_data;
}

// Función auxiliar para encontrar el índice de una columna por nombre
int getColumnIndex(const std::vector<std::string> &headers, const std::string &columnName)
{
    for (int i = 0; i < headers.size(); i++)
    {
        if (headers[i] == columnName)
        {
            return i;
        }
    }
    return -1; // No encontrada
}

// // Ejemplo de uso
// int main() {
//     int rows, cols;
//     std::vector<std::string> headers;

//     float* h_data = readCSV("datos.csv", rows, cols, headers);

//     if (h_data == nullptr) {
//         return -1;
//     }

//     std::cout << "CSV leído: " << rows << " filas, " << cols << " columnas" << std::endl;

//     // Mostrar encabezados
//     std::cout << "Columnas: ";
//     for (const auto& header : headers) {
//         std::cout << header << " ";
//     }
//     std::cout << std::endl;

//     // Ejemplo: obtener índice de una columna específica
//     int colIndex = getColumnIndex(headers, "precio");
//     if (colIndex != -1) {
//         std::cout << "Columna 'precio' está en el índice: " << colIndex << std::endl;

//         // Acceder a valores de esa columna
//         std::cout << "Primeros valores de 'precio': ";
//         for (int i = 0; i < std::min(5, rows); i++) {
//             std::cout << h_data[i * cols + colIndex] << " ";
//         }
//         std::cout << std::endl;
//     }

//     // Usar con CUDA
//     float* d_data;
//     int totalSize = rows * cols;

//     cudaMalloc(&d_data, totalSize * sizeof(float));
//     cudaMemcpy(d_data, h_data, totalSize * sizeof(float), cudaMemcpyHostToDevice);

//     // Tu procesamiento CUDA aquí...

//     // Liberar memoria
//     cudaFree(d_data);
//     delete[] h_data;

//     return 0;
// }