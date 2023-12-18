
#include <vector>

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
void mseDerivative(std::vector<T>& y,  std::vector<T>& target, std::vector<T>& dE_dy);

template<typename T>
std::vector<std::vector<T>> createTempWeightMAtrix(std::vector<std::vector<T>>& old_weights);

template<typename T>
std::vector<std::vector<T>> createTempBiasMAtrix(std::vector<std::vector<T>>& old_bias);

template<typename T>
void updateDE_Dw_Db(std::vector<std::vector<T>>& old_weights, std::vector<std::vector<T>>& old_bias, std::vector<std::vector<T>>& new_weights, std::vector<std::vector<T>>& new_bias);

template<typename T>
void updateWeightsBias(std::vector<std::vector<T>>& old_weights, std::vector<std::vector<T>>& new_weights, std::vector<std::vector<T>>& old_bias, std::vector<std::vector<T>>& new_bias, int numOccurence, float learning_rate);

template<typename T>
void evaluateAccuracy(std::vector<T>& y, std::vector<T>& target, int& numCorrect, int& numTotal);

template<typename T>
void resetVector(std::vector<std::vector<T>>& vector);

//template<typename T>
//void initialiseVector(std::vector<std::vector<T>>& default_weights);

template<typename T>
void incrementweightsBias(std::vector<std::vector<T>>& old_weights, std::vector<std::vector<T>>& old_bias, std::vector<std::vector<T>>& new_weights, std::vector<std::vector<T>>& new_bias);