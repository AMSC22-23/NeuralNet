#ifndef ACTIVATION_FUNCTIONS_HPP
#define ACTIVATION_FUNCTIONS_HPP

#include <cmath>
#include <string>

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

#endif