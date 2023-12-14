
#include <vector>

template<typename T>
std::vector<T> sum(std::vector<T>& a, std::vector<T>& b);

template<typename T>
std::vector<T> transposeMatrix(const std::vector<T>& matrix, const int m, const int n);

template<typename T>
void transposeMatrix2(const std::vector<T>& matrix, std::vector<T>& transposed,  const int m, const int n);