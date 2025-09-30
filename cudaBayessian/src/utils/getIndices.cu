#include <vector>
/// @brief
/// @param data  --> column-major
/// @param rows
/// @param cols
/// @param columnIndex
/// @param targetValue
/// @return
std::vector<int> getIndicesByValue(const float *data, int rows, int cols,
                                   int columnIndex, float targetValue)
{
    std::vector<int> indices;

    // Recordar que estamos en column-major: [columnIndex * rows + i]
    for (int i = 0; i < rows; i++)
    {
        if (data[columnIndex * rows + i] == targetValue)
        {
            indices.push_back(i);
        }
    }

    return indices;
}
