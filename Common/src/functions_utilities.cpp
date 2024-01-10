//
// Created by filippo on 10/01/24.
//
#include "functions_utilities.hpp"


template<typename T>
std::vector<T> operator+(const std::vector<T>& a, const std::vector<T>& b){

    int num_threads;
    int n = a.size();
    std::vector<T> c(n);

#ifdef _OPENMP
    num_threads = 8;
#else
    num_threads = 1;
#endif

#pragma omp parallel for num_threads(num_threads) shared(a, b, c, n)
    for (int i=0; i<n; i++)
        c[i]= a[i] + b[i];

    return c;
}


template std::vector<double> operator+(const std::vector<double>& a, const std::vector<double>& b);
template std::vector<float> operator+(const std::vector<float>& a, const std::vector<float>& b);