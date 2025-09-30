void transposeInPlace(float *data, int rows, int cols)
{
    float *temp = new float[rows * cols];

    // Copiar transponiendo
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            temp[j * rows + i] = data[i * cols + j];
        }
    }

    // Copiar de vuelta
    for (int i = 0; i < rows * cols; i++)
    {
        data[i] = temp[i];
    }

    delete[] temp;
}

// Versión que devuelve nuevo array
float *columnMajorToRowMajor(const float *colMajor, int rows, int cols)
{
    float *rowMajor = new float[rows * cols];

    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            // column-major: [j * rows + i]
            // row-major: [i * cols + j]
            rowMajor[i * cols + j] = colMajor[j * rows + i];
        }
    }

    return rowMajor;
}