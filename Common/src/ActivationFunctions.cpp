#include "ActivationFunctions.hpp"

template <typename T>
T linearActivation(const T &input) {
    return input;
}

template <typename T>
T linearActivationDerivative(const T &input) {
    return T(1);
}

template <typename T>
T sigmoidActivation(const T &input) {
    return 1 / (1 + std::exp(-input));
}

template <typename T>
T sigmoidActivationDerivative(const T &input) {
    T sigmoid = sigmoidActivation(input);
    return sigmoid * (1 - sigmoid);
}

template <typename T>
T tanhActivation(const T &input) {
    return std::tanh(input);
}

template <typename T>
T tanhActivationDerivative(const T &input) {
    T tanhValue = tanhActivation(input);
    return 1 - tanhValue * tanhValue;
}

template <typename T>
T ReLuActivation(const T &input) {
    return input > 0 ? input : 0;
}

template <typename T>
T ReLuActivationDerivative(const T &input) {
    return input > 0 ? 1 : 0;
}