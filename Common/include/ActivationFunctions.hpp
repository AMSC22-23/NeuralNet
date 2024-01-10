#ifndef ACTIVATION_FUNCTIONS_HPP
#define ACTIVATION_FUNCTIONS_HPP

#include <cmath>
#include <string>
#include <vector>

//*********************************************************************************************************************

// Here you can find the declaration of the activation functions, the body is defined in /src/ActivationFunctions.cpp

//*********************************************************************************************************************


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
inline T softmaxActivation(const T &input);

template <typename T>
inline T softmaxActivationDerivative(const T &input);

template <typename T>
T applyActivationFunction(const T &input, const std::string &activationFunction);

template <typename T>
T applyActivationFunction(const T &input, const std::string &activationFunction);

template<typename T>
T applyActivationFunctionDerivative(const T &input, const std::string &activationFunction);





#endif