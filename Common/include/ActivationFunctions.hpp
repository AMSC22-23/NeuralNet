#ifndef ACTIVATION_FUNCTIONS_HPP
#define ACTIVATION_FUNCTIONS_HPP

#include <cmath>
#include <string>
#include <vector>

template <typename T>
inline T linearActivation(const T &input);

template <typename T>
inline T linearActivationDerivative(const T &input);

template <typename T>
inline T sigmoidActivation(const T &input);

template <typename T>
inline T sigmoidActivationDerivative(const T &input);

template <typename T>
inline T tanhActivation(const T &input);

template <typename T>
inline T tanhActivationDerivative(const T &input);

template <typename T>
inline T ReLuActivation(const T &input);

template <typename T>
inline T ReLuActivationDerivative(const T &input);

template <typename T>
T applyActivationFunction(const T &input, const std::string &activationFunction);

template <typename T>
T applyActivationFunction(const T &input, const std::string &activationFunction);

template<typename T>
T applyActivationFunctionDerivative(const T &input, const std::string &activationFunction);

template<typename T>
void mseDerivative(std::vector<T>& y, std::vector<T>& target, std::vector<T>& dE_dy);

//@note: it is worth it to pass an object by reference if its size is less than one word
// of memory, for ints of normal laptops you usually have that sizeof(int) == 4 and one 
// word of memory has size 8, so it is better to pass by copy
template<typename T>
std::vector<T> transposeMatrix(std::vector<T>& matrix, int& m, int& n);

//@note: should pass the string by const reference
template<typename T>
void applyLossFunction(std::vector<T>& y, std::vector<T>& target, std::vector<T>& dE_dy, std::string& lossFunction);

//template<typename T>
//std::vector<T> transposeMatrix(const std::vector<T>& matrix, const int m, const int n);



#endif