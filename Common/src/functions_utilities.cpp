//
// Created by filippo on 10/01/24.
//
#include "functions_utilities.hpp"
#include <cmath>

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


template<typename T>
T mse(const std::vector<T>& y, const std::vector<T>& target, int num_threads) {

#ifdef _OPENMP
    omp_set_num_threads(num_threads);
#endif

    T result = 0;
#pragma omp parallel for shared(y, target) reduction(+:result)
    for(int i = 0; i < y.size(); i++){
        result += pow(y[i] - target[i], 2);
    }
    result = result / y.size();
    return result;


}


template float mse<float>(const std::vector<float>& y,const std::vector<float>& target, int num_threads);
template double mse<double>(const std::vector<double>& y,const std::vector<double>& target, int num_threads);