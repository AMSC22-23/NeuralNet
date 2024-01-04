#ifndef UTILFUNCTIONS_HPP
#define UTILFUNCTIONS_HPP

//DEFINED IN functions_utilities.cpp in this folder


#include <vector>

//@note: it is completely fine to have functions like this one but consider
//       1. it might be the case to implement a operator+() which is more readable
//       2. operations like sum(sum(a, b), c) might quickly become expesive, since you
//          need a for loop for each sum. A possible solution is template expressions
//@note: since `a` and `b` should not change, it would be better to add `const`
// these two problems are not limited to here
template<typename T>
std::vector<T> sum(std::vector<T>& a, std::vector<T>& b);

template<typename T>
std::vector<T> transposeMatrix(const std::vector<T>& matrix, const int m, const int n);

template<typename T>
void transposeMatrix2(const std::vector<T>& matrix, std::vector<T>& transposed,  const int m, const int n);

template<typename T>
void mseDerivative(std::vector<T>& y, std::vector<T>& target, std::vector<T>& dE_dy);

template<typename T>
void applyLossFunction( std::vector<T>& y, std::vector<T>& target, std::vector<T>& dE_dy, std::string& lossFunction);

template<typename T>
T evaluateLossFunction(std::vector<T>& y, std::vector<T>& target, std::string& lossFunction);

template<typename T>
T mse(std::vector<T>& y, std::vector<T>& target);

//template<typename T>
//void mseDerivative(std::vector<T>& y,  std::vector<T>& target, std::vector<T>& dE_dy);

template<typename T>
std::vector<std::vector<T>> createTempWeightMAtrix(std::vector<std::vector<T>>& old_weights);

template<typename T>
std::vector<std::vector<T>> createTempBiasMAtrix(std::vector<std::vector<T>>& old_bias);

template<typename T>
void updateDE_Dw_Db(std::vector<std::vector<T>>& old_weights, std::vector<std::vector<T>>& old_bias, std::vector<std::vector<T>>& new_weights, std::vector<std::vector<T>>& new_bias);

template<typename T>
void updateWeightsBias(std::vector<std::vector<T>>& old_weights, std::vector<std::vector<T>>& new_weights, std::vector<std::vector<T>>& old_bias, std::vector<std::vector<T>>& new_bias, int numOccurence, float learning_rate);

//template<typename T>
//void updateWeightsBias(std::vector<std::vector<T>>& old_weights, std::vector<std::vector<T>>& new_weights, T *old_bias, T *new_bias, int numOccurence, float learning_rate);

template<typename T>
void evaluateAccuracy(std::vector<T>& y, std::vector<T>& target, int& numCorrect, int& numTotal);

template<typename T>
void resetVector(std::vector<std::vector<T>>& vector);

//template<typename T>
//void initialiseVector(std::vector<std::vector<T>>& default_weights);

template<typename T>
void incrementweightsBias(std::vector<std::vector<T>>& old_weights, std::vector<std::vector<T>>& old_bias, std::vector<std::vector<T>>& new_weights, std::vector<std::vector<T>>& new_bias);

template<typename T>
void shuffleData(std::vector<std::vector<T>>& trainSet, std::vector<std::vector<T>>& trainOut);

template<typename T>
void mul_funct(std::vector<T>& a, std::vector<T>& b, std::vector<T>& c, int m, int n, int nb, int selection);

template<typename T>
void mul_funct(std::vector<T>& a, std::vector<T>& b, std::vector<T>& c, int m, int n, int nb, int selection, int block_size);

//template<typename T>
//void mul_funct(T *a, T *b, T *c, int m, int n, int nb, int selection, int block_size);



#endif