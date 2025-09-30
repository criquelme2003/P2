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
