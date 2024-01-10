#include "ActivationFunctions.hpp"
#include <iostream>

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
    return (input > 0) ? input : 0;
}

template <typename T>
T ReLuActivationDerivative(const T &input) {
    return (input > 0) ? 1 : 0;
}

//@note: instead of this list of `if` you could either
//       - use polymorphism and have a base class "ActivationFunction" that must 
//         implement a "eval" and "eval_der" methods
//       - this can be done also with static polymorphism if you template upon the
//         activation function type
//       - have a struct "ActivationFunction" that has two attibutes that are std::functions
template <typename T>
T applyActivationFunction(const T &input, const std::string &activationFunction) {
    if (activationFunction == "linear") {
        return linearActivation(input);
    } else if (activationFunction == "sigmoid") {
        return sigmoidActivation(input);
    } else if (activationFunction == "tanh") {
        return tanhActivation(input);
    } else if (activationFunction == "ReLu") {
        return ReLuActivation(input);
    } else {
        std::cout << "Activation function not implemented" << std::endl;
    }
    return 0;
}

template float applyActivationFunction<float>(const float &input, const std::string &activationFunction);
template double applyActivationFunction<double>(const double &input, const std::string &activationFunction);

template<typename T>
T applyActivationFunctionDerivative(const T &input, const std::string &activationFunction) {
    if (activationFunction == "linear") {
        return linearActivationDerivative(input);
    } else if (activationFunction == "sigmoid") {
        return sigmoidActivationDerivative(input);
    } else if (activationFunction == "tanh") {
        return tanhActivationDerivative(input);
    } else if (activationFunction == "ReLu") {
        return ReLuActivationDerivative(input);
    } else {
        std::cout << "Activation function not implemented" << std::endl;
    }
    return 0;
}

template float applyActivationFunctionDerivative<float>(const float &input, const std::string &activationFunction);
template double applyActivationFunctionDerivative<double>(const double &input, const std::string &activationFunction);