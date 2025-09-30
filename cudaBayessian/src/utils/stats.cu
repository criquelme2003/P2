#include <iostream>

void printClassProportions(const float *data, int rows, int targetColumnIndex,
                           const std::string &setName)
{
    int pos_count = 0;
    int neg_count = 0;

    for (int i = 0; i < rows; i++)
    {
        if (data[targetColumnIndex * rows + i] == 1.0f)
        {
            pos_count++;
        }
        else
        {
            neg_count++;
        }
    }

    std::cout << "\nProporción en " << setName << ":" << std::endl;
    std::cout << "  pos: " << static_cast<float>(pos_count) / rows << std::endl;
    std::cout << "  neg: " << static_cast<float>(neg_count) / rows << std::endl;
}