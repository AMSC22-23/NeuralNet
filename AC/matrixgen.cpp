#include <vector>
#include <cstring>
#include <ctime>
#include "matrixgen.hpp"

template <typename T>
std::vector<std::vector<T>> matrixgen(int m, int n){
    std::srand(static_cast<unsigned>(std::time(0)));
    std::vector<std::vector<T>> matrix(m, std::vector<T>(n));
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            matrix[i][j] = static_cast<T>(std::rand()) / static_cast<T>(RAND_MAX) * 100.0;
        }
    }

    return matrix;

    
    

    



}