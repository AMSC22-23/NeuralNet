#ifndef UTILFUNCTIONS_HPP
#define UTILFUNCTIONS_HPP

//**********************************************************************************************************************

//Here you can find the declaration of the functions used in the network.
//You can find the body and the definition of the functions in /src/network_functions.cpp

//**********************************************************************************************************************

#include <string>
#include <vector>

template<typename T>
std::vector<T> sum(const std::vector<T>& a,const std::vector<T>& b);

template<typename T>
std::vector<T> operator+(const std::vector<T>& a, const std::vector<T>& b);

template<typename T>
std::vector<T> transposeMatrix(const std::vector<T>& matrix, const int m, const int n);

template<typename T>
void transposeMatrix2(const std::vector<T>& matrix, std::vector<T>& transposed,  const int m, const int n);

template<typename T>
void mseDerivative(const std::vector<T>& y,const  std::vector<T>& target, std::vector<T>& dE_dy);

template<typename T>
void applyLossFunction( const std::vector<T>& y, const std::vector<T>& target, std::vector<T>& dE_dy, const std::string& lossFunction);

template<typename T>
T evaluateLossFunction(const std::vector<T>& y, const std::vector<T>& target, const std::string& lossFunction);

template<typename T>
T mse(const std::vector<T>& y, const std::vector<T>& target);

template<typename T>
T mse(const std::vector<T>& y, const std::vector<T>& target, int num_threads);

template<typename T>
void mseDerivative(const std::vector<T>& y,  const std::vector<T>& target, std::vector<T>& dE_dy);

template<typename T>
std::vector<std::vector<T>> createTempWeightMAtrix(const std::vector<std::vector<T>>& old_weights);

template<typename T>
std::vector<std::vector<T>> createTempBiasMAtrix(const std::vector<std::vector<T>>& old_bias);

template<typename T>
void updateDE_Dw_Db(const std::vector<std::vector<T>>& old_weights, const std::vector<std::vector<T>>& old_bias, std::vector<std::vector<T>>& new_weights, std::vector<std::vector<T>>& new_bias);

template<typename T>
void updateWeightsBias(std::vector<std::vector<T>>& old_weights, std::vector<std::vector<T>>& new_weights, std::vector<std::vector<T>>& old_bias, std::vector<std::vector<T>>& new_bias, int numOccurence, float learning_rate);

//template<typename T>
//void updateWeightsBias(std::vector<std::vector<T>>& old_weights, std::vector<std::vector<T>>& new_weights, T *old_bias, T *new_bias, int numOccurence, float learning_rate);

//template<typename T>
//void evaluateAccuracy(const std::vector<T>& y, const std::vector<T>& target, int& numCorrect, int& numTotal);

template<typename T>
void resetVector(std::vector<std::vector<T>>& vector);

//template<typename T>
//void initialiseVector(std::vector<std::vector<T>>& default_weights);

template<typename T>
void incrementweightsBias(const std::vector<std::vector<T>>& old_weights, const std::vector<std::vector<T>>& old_bias, std::vector<std::vector<T>>& new_weights, std::vector<std::vector<T>>& new_bias);

template<typename T>
void shuffleData(std::vector<std::vector<T>>& trainSet, std::vector<std::vector<T>>& trainOut);

template<typename T>
void mul_funct(std::vector<T>& a, std::vector<T>& b, std::vector<T>& c, int m, int n, int nb, int selection);

template<typename T>
void mul_funct(std::vector<T>& a, std::vector<T>& b, std::vector<T>& c, int m, int n, int nb, int selection, int block_size);

//template<typename T>
//void mul_funct(T *a, T *b, T *c, int m, int n, int nb, int selection, int block_size);



#endif