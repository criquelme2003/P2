#include <iostream>
#include <random>
float *readCSV(const std::string &filename, int &rows, int &cols,
               std::vector<std::string> &headers, char delimiter = ',',
               bool hasHeader = true);
void transposeInPlace(float *data, int rows, int cols);

std::vector<int> getIndicesByValue(const float *data, int rows, int cols,
                                   int columnIndex, float targetValue);

void stratifiedSplit(const float *data, int rows, int cols,
                     int targetColumnIndex,
                     std::vector<int> &train_indices,
                     std::vector<int> &test_indices,
                     float trainRatio = 0.8f,
                     unsigned int seed = 123);

float *createSubset(const float *data, int rows, int cols,
                    const std::vector<int> &indices);

void printClassProportions(const float *data, int rows, int targetColumnIndex,
                           const std::string &setName);

float *filterByValue(const float *data, int rows, int cols,
                     int columnIndex, float targetValue, int &newRows);

float *columnMajorToRowMajor(const float *colMajor, int rows, int cols);

void bootstrapSample(int n_samples,
                     std::vector<int> &bootstrap_indices,
                     std::vector<int> &oob_indices,
                     std::mt19937 &rng);