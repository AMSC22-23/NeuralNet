#include "model.hpp"
#include "matrixProd_VM_VV.hpp"
#include "ActivationFunctions.hpp"
#include "functions_utilities.hpp"

template<typename T>
std::vector<T> sum(std::vector<T>& a, std::vector<T>& b){
    std::vector<T> c;
    c.resize(a.size());
    for(int i = 0; i < a.size(); i++){
        c[i] = a[i] + b[i];
    }
    return c;
}
template std::vector<float> sum<float>(std::vector<float>& a, std::vector<float>& b);
template std::vector<double> sum<double>(std::vector<double>& a, std::vector<double>& b);

template<typename T>
std::vector<T> mul(std::vector<T>& a, std::vector<T>& b){
    std::vector<T> c;
    c.resize(a.size());
    for(int i = 0; i < a.size(); i++){
        c[i] = a[i] * b[i];
    }
    return c;
}
template std::vector<float> mul<float>(std::vector<float>& a, std::vector<float>& b);
template std::vector<double> mul<double>(std::vector<double>& a, std::vector<double>& b);

template<typename T>
void Model<T>::extendMatrix(){
    //input.push_back(1);
    weights[0].resize(weights_shape[0][0]+bias[0].size());
    weights[0].insert(weights[0].end(), bias[0].begin(), bias[0].end());
    for(int loop = 0; loop < layers.size(); loop++){
        h[loop].push_back(1);
        weights[loop+1].resize(weights_shape[loop+1][0]+bias[loop+1].size());
        weights[loop+1].insert(weights[loop+1].end(), bias[loop+1].begin(), bias[loop+1].end());
    }
}

template void Model<float>::extendMatrix();
template void Model<double>::extendMatrix();

template<typename T>
void Model<T>::reduceMatrix(){
    //input.pop_back();
    //weights[0].erase(weights[0].end()-bias[0].size(), weights[0].end());
    weights[0].resize(weights_shape[0][0]);
    for(int loop = 0; loop < layers.size(); loop++){
        h[loop].pop_back();
        //weights[loop+1].erase(weights[loop+1].end()-bias[loop+1].size(), weights[loop+1].end());
        weights[loop+1].resize(weights_shape[loop+1][0]);
    }
}

template void Model<float>::reduceMatrix();
template void Model<double>::reduceMatrix();

template<typename T>
void mul_funct(std::vector<T>& a, std::vector<T>& b, std::vector<T>& c, int m, int n, int nb, int selection){
    int64_t t;
    switch(selection){
        case 0:
            MatrixCaheOptimised<T>(a, b, c, m, n, nb, t);
            break;
    }    
}

template void mul_funct<float>(std::vector<float>& a, std::vector<float>& b, std::vector<float>& c, int m, int n, int nb, int selection);
template void mul_funct<double>(std::vector<double>& a, std::vector<double>& b, std::vector<double>& c, int m, int n, int nb, int selection);

template<typename T>
void activationFun(std::vector<T>& a, std::vector<T>& b, std::string activation){
    std::cout << "Passata activation function: " << activation << std::endl;
    for(int i = 0; i < a.size(); i++){
        b[i] = applyActivationFunction(a[i], activation);
    }
}

template void activationFun<float>(std::vector<float>& a, std::vector<float>& b, std::string activation);
template void activationFun<double>(std::vector<double>& a, std::vector<double>& b, std::string activation);

template<typename T>
void activationFunDerivative(std::vector<T>& a, std::vector<T>& b, std::string activation){
    std::cout << "Passata activation function: " << activation << std::endl;
    for(int i = 0; i < a.size(); i++){
        b[i] = applyActivationFunctionDerivative(a[i], activation);
    }
}
template void activationFunDerivative<float>(std::vector<float>& a, std::vector<float>& b, std::string activation);
template void activationFunDerivative<double>(std::vector<double>& a, std::vector<double>& b, std::string activation);

template<typename T> //thi version need to be called only after the resizing of the weights
void Model<T>::predict(std::vector<T>& input, int& selection){
    input.push_back(1);
    //weights[0].resize(weights_shape[0][0]+bias[0].size());
    //weights[0].insert(weights[0].end(), bias[0].begin(), bias[0].end());
    mul_funct(input, weights[0], z[0], 1, weights_shape[0][0]+1, weights_shape[0][1], matrix_mul_optimisation);   //weights[0], input, z, weights_shape[0][0], weights_shape[0][1], 1, matrix_mul_optimisation);
    activationFun(z[0], h[0], layers[0].getActFun());
    
    for(int loop = 0; loop < layers.size(); loop++){
        //h[loop].push_back(1);
        //weights[loop+1].resize(weights_shape[loop+1][0]+bias[loop+1].size());
        //weights[loop+1].insert(weights[loop+1].end(), bias[loop+1].begin(), bias[loop+1].end());
        mul_funct(h[loop], weights[loop+1], z[loop+1], 1, weights_shape[loop+1][0]+1, weights_shape[loop+1][1], matrix_mul_optimisation);
        if(loop < layers.size()-1){
            activationFun(z[loop+1], h[loop+1], layers[loop+1].getActFun());
        }
    }
    activationFun(z[layers.size()], y, model_output.getOutputAct_fun());
    std::cout << "output: " << std::endl;
    for(int i = 0; i < y.size(); i++){
        std::cout << y[i] << " ";
    }
    std::cout << std::endl;
    input.pop_back();
}

template void Model<float>::predict(std::vector<float>& input, int& selection);
template void Model<double>::predict(std::vector<double>& input, int& selection);

template<typename T> //this version contains the extension and reduction of the matrix
void Model<T>::predict(std::vector<T>& input, int& selection, int flag){
    extendMatrix();
    input.push_back(1);
    //weights[0].resize(weights_shape[0][0]+bias[0].size());
    //weights[0].insert(weights[0].end(), bias[0].begin(), bias[0].end());
    mul_funct(input, weights[0], z[0], 1, weights_shape[0][0]+1, weights_shape[0][1], matrix_mul_optimisation);   //weights[0], input, z, weights_shape[0][0], weights_shape[0][1], 1, matrix_mul_optimisation);
    activationFun(z[0], h[0], layers[0].getActFun());
    
    for(int loop = 0; loop < layers.size(); loop++){
        //h[loop].push_back(1);
        //weights[loop+1].resize(weights_shape[loop+1][0]+bias[loop+1].size());
        //weights[loop+1].insert(weights[loop+1].end(), bias[loop+1].begin(), bias[loop+1].end());
        mul_funct(h[loop], weights[loop+1], z[loop+1], 1, weights_shape[loop+1][0]+1, weights_shape[loop+1][1], matrix_mul_optimisation);
        if(loop < layers.size()-1){
            activationFun(z[loop+1], h[loop+1], layers[loop+1].getActFun());
        }
    }
    activationFun(z[layers.size()], y, model_output.getOutputAct_fun());
    std::cout << "output: " << std::endl;
    for(int i = 0; i < y.size(); i++){
        std::cout << y[i] << " ";
    }
    std::cout << std::endl;
    input.pop_back();
    reduceMatrix();
}

template void Model<float>::predict(std::vector<float>& input, int& selection, int flag);
template void Model<double>::predict(std::vector<double>& input, int& selection, int flag);


//note that this function is defined only in this file, so it is not necessary to declare it in the header file
template<typename T>
std::vector<T> transposeMatrix(const std::vector<T>& matrix, const int m, const int n){
    std::vector<T> transposed_matrix;
    transposed_matrix.resize(1);
    transposed_matrix.resize(m*n);
    for(int i = 0; i < m; i++){
        for(int j = 0; j < n; j++){
            transposed_matrix[j+i*n] = matrix[i+j*m];
        }
    }
    return transposed_matrix;
}
template std::vector<float> transposeMatrix(const std::vector<float>& matrix, const int m, const int n);
template std::vector<double> transposeMatrix(const std::vector<double>& matrix, const int m, const int n);

template<typename T>
void transposeMatrix2(const std::vector<T>& matrix, std::vector<T>& transposed,  const int m, const int n){
    transposed.resize(1);
    transposed.resize(m*n);
    for(int i=0; i<m; i++){
        for(int j=0; j<n; j++){
            transposed[j+i*n] = matrix[i+j*m];
        }
    }
}
template void transposeMatrix2<float>(const std::vector<float>& matrix, std::vector<float>& transposed,  const int m, const int n);
template void transposeMatrix2<double>(const std::vector<double>& matrix, std::vector<double>& transposed,  const int m, const int n);


//defined in activationFunctions.hpp
template<typename T>
void mseDerivative(std::vector<T>& y, std::vector<T>& target, std::vector<T>& dE_dy){
    for(int i = 0; i < y.size(); i++){
        dE_dy[i] = y[i] - target[i];
    }
}
template void mseDerivative<float>(std::vector<float>& y, std::vector<float>& target, std::vector<float>& dE_dy);
template void mseDerivative<double>(std::vector<double>& y, std::vector<double>& target, std::vector<double>& dE_dy);

template<typename T>
void applyLossFunction(std::vector<T>& y, std::vector<T>& target, std::vector<T>& dE_dy, std::string& lossFunction){
    if(lossFunction == "MSE"){
        mseDerivative(y, target, dE_dy);
        std::cout << "dE_dy: " <<std::endl;
        for(int i=0; i<dE_dy.size(); i++){
            std::cout << dE_dy[i] << " ";
        }
        std::cout <<std::endl;
    }
}
template void applyLossFunction<float>(std::vector<float>& y, std::vector<float>& target, std::vector<float>& dE_dy, std::string& lossFunction);
template void applyLossFunction<double>(std::vector<double>& y, std::vector<double>& target, std::vector<double>& dE_dy, std::string& lossFunction);

template<typename T>
void Model<T>::backPropagation(std::vector<T>& input, std::vector<T>& dE_dy, int& selection){
    int one=1;
    std::vector<T> temp;
    activationFunDerivative(z[layers.size()], dAct_z[layers.size()], model_output.getOutputAct_fun());
    dE_db[layers.size()] = mul(dE_dy, dAct_z[layers.size()]);
    temp = transposeMatrix(h[layers.size()-1], one, h[layers.size()-1].size());
    //mul_funct(temp , dE_db[layers.size()], dE_dw[layers.size()], dim1, one, dim2, matrix_mul_optimisation);
    mul_funct(temp , dE_db[layers.size()],dE_dw[layers.size()], h[layers.size()-1].size(), one, dE_db[layers.size()].size(), matrix_mul_optimisation);
    //temp = transposeMatrix(weights[layers.size()], weights_shape[layers.size()][0], weights_shape[layers.size()][1]);
    transposeMatrix2(weights[layers.size()], temp, weights_shape[layers.size()][0], weights_shape[layers.size()][1]);
    mul_funct(dE_db[layers.size()], temp, dE_dx[layers.size()], one,  dE_db[layers.size()].size(), weights_shape[layers.size()][0], matrix_mul_optimisation);
    for (int i=layers.size()-1; i > 0; i--){
        activationFunDerivative(z[i], dAct_z[i], layers[i].getActFun());
        dE_db[i] = mul(dE_dx[i], dAct_z[i]);
        temp = transposeMatrix(h[i-1], one, h[i-1].size());
        mul_funct(temp, dE_db[i], dE_dw[i], h[i-1].size(), one, dE_db[i].size(), matrix_mul_optimisation);
        //temp = transposeMatrix(weights[i], weights_shape[i][0], weights_shape[i][1]);
        transposeMatrix2(weights[i], temp, weights_shape[i][0], weights_shape[i][1]);
        mul_funct(dE_db[i], temp, dE_dx[i], one,  dE_db[i].size(), weights_shape[i][0], matrix_mul_optimisation);
    }
    activationFunDerivative(z[0], dAct_z[0], layers[0].getActFun());
    dE_db[0] = mul(dE_dx[0], dAct_z[0]);
    temp = transposeMatrix(input, one, input.size());
    mul_funct(temp, dE_db[0], dE_dw[0], input.size(), one, dE_db[0].size(), matrix_mul_optimisation);
    std::cout << "dE_dw[0]: " << std::endl;
    for(int s = 0; s < weights_shape[0][0]; s++){
        for(int t = 0; t < weights_shape[0][1]; t++){
            std::cout << dE_dw[0][t+s*weights_shape[0][1]] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
    std::cout << "dE_dw[1]: " << std::endl;
    for(int s = 0; s < weights_shape[1][0]; s++){
        for(int t = 0; t < weights_shape[1][1]; t++){
            std::cout << dE_dw[1][t+s*weights_shape[1][1]] << " ";
        }
        std::cout << std::endl;
    }
}
template void Model<float>::backPropagation(std::vector<float>& input, std::vector<float>& dE_dy, int& selection);
template void Model<double>::backPropagation(std::vector<double>& input, std::vector<double>& dE_dy, int& selection);

