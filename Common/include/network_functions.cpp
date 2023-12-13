#include "model.hpp"
#include "matrixProd_VM_VV.hpp"
#include "ActivationFunctions.hpp"

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

template<typename T>
void Model<T>::backPropagation(std::vector<T>& input, std::vector<T>& dE_dy, int& selection){
    
}