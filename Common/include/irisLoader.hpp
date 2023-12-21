#ifndef IRISLOADER_HPP
#define IRISLOADER_HPP

#include<tuple>
#include<vector>
#include<string>

template<typename T>
void genFakeData(std::vector<std::vector<T>>& a, int rows, int cols);

template<typename T>
T normalize(T value, T min, T max);

template<typename T>
std::tuple<std::vector<std::tuple<T, T, T, T, std::tuple<int, int, int>>>, std::vector<std::tuple<T, T, T, T, std::tuple<int, int, int>>>, std::vector<std::tuple<T, T, T, T, std::tuple<int, int, int>>>> readIrisData(const std::string& file_name);


template<typename T>
std::tuple<std::vector<std::vector<T>>, std::vector<std::vector<T>>,
           std::vector<std::vector<T>>, std::vector<std::vector<T>>,
           std::vector<std::vector<T>>, std::vector<std::vector<T>>>
getIrisSets(const std::tuple<std::vector<std::tuple<T, T, T, T, std::tuple<int, int, int>>>, std::vector<std::tuple<T, T, T, T, std::tuple<int, int, int>>>, std::vector<std::tuple<T, T, T, T, std::tuple<int, int, int>>>>& iris_data,
          double train_ratio, double val_ratio, double test_ratio);


#endif