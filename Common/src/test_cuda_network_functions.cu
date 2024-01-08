#include <cuda.h>
#include <cuda_runtime_api.h>
#include "../include/test_model.hpp"
#include "../include/matrixProd_VM_VV.hpp"
#include "../include/ActivationFunctions.hpp"
#include "../include/functions_utilities.hpp"
#include "../include/matrixProd_AVX.hpp"
#include "../include/cudaMatrixMul.h"
#include <algorithm>
#include <random>
#include <iomanip>
#include <fstream>
#include<chrono>



/*
 * ***********************************************************************************************************************
 * *******************************    UTILITIES **************************************************************************
 * ***********************************************************************************************************************
*/

//***************************************************************************************************************************
//This finction tacke as input the train and the output data and shuffle them

template<typename T>
void shuffleData(std::vector<std::vector<T>>& firstSet, std::vector<std::vector<T>>& secondSet){
    if(firstSet.size() != secondSet.size()){
        std::cout << "Error: the two set have different size" << std::endl;
        return;
    }
    std::vector<std::vector<T>> temp1, temp2;
    std::vector<int> index;
    //std::random_device rd;  //aumento la qualità random ma seed non fisso
    //std::mt19937 g(rd());
    std::mt19937 g(44); //seed fisso
    index.resize(firstSet.size());
    std::iota(index.begin(), index.end(), 0);
    std::shuffle(index.begin(), index.end(), g);

    temp1.resize(firstSet.size());
    temp2.resize(firstSet.size());

    for(int i = 0; i < firstSet.size(); i++){
        temp1[i].resize(firstSet[i].size());
        temp2[i].resize(secondSet[i].size());
        temp1[i] = firstSet[index[i]];
        temp2[i] = secondSet[index[i]];
    }
    firstSet = temp1;
    secondSet = temp2;
    
}

template void shuffleData<float>(std::vector<std::vector<float>>& firstSet, std::vector<std::vector<float>>& secondSet);
template void shuffleData<double>(std::vector<std::vector<double>>& firstSet, std::vector<std::vector<double>>& secondSet);


//***************************************************************************************************************************
//Perform an increment sum of weight and bias matrix, used in train() function


template<typename T>
void incrementweightsBias(std::vector<std::vector<T>>& old_weights, std::vector<std::vector<T>>& old_bias, std::vector<std::vector<T>>& new_weights, std::vector<std::vector<T>>& new_bias){
    for(int i=0; i<old_weights.size(); i++){
        for(int j=0; j<old_weights[i].size(); j++){
            new_weights[i][j] = old_weights[i][j] + new_weights[i][j];
        }
        for(int j=0; j<old_bias[i].size(); j++){
            new_bias[i][j] = old_bias[i][j] + new_bias[i][j];
        }
    }
}
template void incrementweightsBias<float>(std::vector<std::vector<float>>& old_weights, std::vector<std::vector<float>>& old_bias, std::vector<std::vector<float>>& new_weights, std::vector<std::vector<float>>& new_bias);
template void incrementweightsBias<double>(std::vector<std::vector<double>>& old_weights, std::vector<std::vector<double>>& old_bias, std::vector<std::vector<double>>& new_weights, std::vector<std::vector<double>>& new_bias);



//****************************************************************************************************************************************************
//This function set to 0 each element of an undefined vector, used to reset the temporary matrix used to store the derivatives of the loss function with respect to the weights and the bias

template<typename T>
void resetVector(std::vector<std::vector<T>>& vector){
    for (auto& row : vector) {
        std::fill(row.begin(), row.end(), 0.0);
    }
}
template void resetVector<float>(std::vector<std::vector<float>>& vector);
template void resetVector<double>(std::vector<std::vector<double>>& vector);

//****************************************************************************************************************************************************
//DEPRECATED
template<typename T>
void evaluateAccuracy(std::vector<T>& y, std::vector<T>& target, int& numCorrect, int& numTotal){
    for(int i = 0; i < y.size(); i++){
        if(y[i] == target[i]){
            numCorrect++;
        }
        numTotal++;
    }
}

//****************************************************************************************************************************************************
//These functions are used to update the temporary Matrix used to store the derivatives of the loss function with respect to the weights and the bias

template<typename T>
void updateWeightsBias(std::vector<std::vector<T>>& old_weights, std::vector<std::vector<T>>& new_weights, std::vector<std::vector<T>>& old_bias, std::vector<std::vector<T>>& new_bias, int numOccurence, float learning_rate){
    for(int i=0; i<old_weights.size(); i++){
        for(int j=0; j<old_weights[i].size(); j++){
            old_weights[i][j] = old_weights[i][j] - learning_rate * new_weights[i][j] / numOccurence;
        }
        for(int j=0; j<old_bias[i].size(); j++){
            old_bias[i][j] = old_bias[i][j] - learning_rate * new_bias[i][j] / numOccurence;
        }
    }
}
template void updateWeightsBias<float>(std::vector<std::vector<float>>& old_weights, std::vector<std::vector<float>>& new_weights, std::vector<std::vector<float>>& old_bias, std::vector<std::vector<float>>& new_bias, int numOccurence, float learning_rate);
template void updateWeightsBias<double>(std::vector<std::vector<double>>& old_weights, std::vector<std::vector<double>>& new_weights, std::vector<std::vector<double>>& old_bias, std::vector<std::vector<double>>& new_bias, int numOccurence, float learning_rate);


template<typename T>
void updateDE_Dw_Db(std::vector<std::vector<T>>& old_weights, std::vector<std::vector<T>>& old_bias, std::vector<std::vector<T>>& new_weights, std::vector<std::vector<T>>& new_bias){
    for(int i=0; i<old_weights.size(); i++){
        new_weights[i] = sum(old_weights[i], new_weights[i]);
        new_bias[i] = sum(old_bias[i], new_bias[i]);
    }
}
template void updateDE_Dw_Db<float>(std::vector<std::vector<float>>& old_weights, std::vector<std::vector<float>>& old_bias, std::vector<std::vector<float>>& new_weights, std::vector<std::vector<float>>& new_bias);
template void updateDE_Dw_Db<double>(std::vector<std::vector<double>>& old_weights, std::vector<std::vector<double>>& old_bias, std::vector<std::vector<double>>& new_weights, std::vector<std::vector<double>>& new_bias);

//****************************************************************************************************************************************************
//Used to initialize to 0 objects with the same dimensions of vector of weights and bias, used to create temporary matrix for the training


template<typename T>
std::vector<std::vector<T>> createTempBiasMAtrix(std::vector<std::vector<T>>& old_bias){
    std::vector<std::vector<T>> temp;
    temp.resize(old_bias.size());
    for(int i = 0; i < old_bias.size(); i++){
        temp[i].resize(old_bias[i].size());
        for(int j = 0; j < old_bias[i].size(); j++){
            //temp[i][j] = old_bias[i][j];
            temp[i][j] = 0.0;
        }
    }
    return temp;
}
template std::vector<std::vector<float>> createTempBiasMAtrix(std::vector<std::vector<float>>& old_bias);
template std::vector<std::vector<double>> createTempBiasMAtrix(std::vector<std::vector<double>>& old_bias);

template<typename T>  //da ottimizzare
std::vector<std::vector<T>> createTempWeightMAtrix(std::vector<std::vector<T>>& old_weights){
    std::vector<std::vector<T>> temp;
    temp.resize(old_weights.size());
    for(int i = 0; i < old_weights.size(); i++){
        temp[i].resize(old_weights[i].size());
        for(int j = 0; j < old_weights[i].size(); j++){
            //temp[i][j] = old_weights[i][j];
            temp[i][j] = 0.0;
        }
    }
    return temp;
}
template std::vector<std::vector<float>> createTempWeightMAtrix(std::vector<std::vector<float>>& old_weights);
template std::vector<std::vector<double>> createTempWeightMAtrix(std::vector<std::vector<double>>& old_weights);

//****************************************************************************************************************************************************
//Perform the sum (first) and multiplication (second) of members with the same index of two different matrix stored in two different row-major vectors


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



//****************************************************************************************************************************************************
/**
 * This function take two vector as input and store the result of the MAtrix multiplication in a third vector, several optimization are available
 * modify the variable matrix_mul_optimisation to select the optimization:
 *      0) GPU with block size optimization (default)
 *      
 * 
 * the other parameters are:
 *     a: first matrix
 *     b: second matrix
 *     c: result matrix
 *     m: number of rows of the first matrix
 *     n: number of columns of the first matrix
 *     nb: number of columns of the second matrix
*/

template<>
void mul_funct<float>(std::vector<float>& a, std::vector<float>& b, std::vector<float>& c, int m, int n, int nb, int selection, int block_size){
    float *ac,*bc,*cc;
    ac = a.data();
    bc = b.data();
    cc = c.data();
    //int cuda_block_size = 32;
    switch(selection){
        case 0:
            cudaFunctionF(ac, bc, cc, m, n, nb, block_size);
            break;

        case 1:
            cudaTileFunctionF(ac, bc, cc, m, n, nb, block_size);
            break;
    }    
}

template<>
void mul_funct<double>(std::vector<double>& a, std::vector<double>& b, std::vector<double>& c, int m, int n, int nb, int selection, int block_size){
    double *ac,*bc,*cc;
    ac = a.data();
    bc = b.data();
    cc = c.data();
    //int cuda_block_size = 32;
    switch(selection){
        case 0:
            cudaFunctionD(ac, bc, cc, m, n, nb, block_size);
            break;

        case 1:
            cudaTileFunctionD(ac, bc, cc, m, n, nb, block_size);
            break;

    }
}


//template void mul_funct<float>(std::vector<float>& a, std::vector<float>& b, std::vector<float>& c, int m, int n, int nb, int selection);
//template void mul_funct<double>(std::vector<double>& a, std::vector<double>& b, std::vector<double>& c, int m, int n, int nb, int selection);


//****************************************************************************************************************************************************
/**
 * These two functions redirect the reference of the input to be evaluated to the correct activation function or its derivative
*/

template<typename T>
void activationFun(std::vector<T>& a, std::vector<T>& b, std::string activation){
    //std::cout << "Passata activation function: " << activation << std::endl;
    for(int i = 0; i < a.size(); i++){
        b[i] = applyActivationFunction(a[i], activation);
    }
}

template void activationFun<float>(std::vector<float>& a, std::vector<float>& b, std::string activation);
template void activationFun<double>(std::vector<double>& a, std::vector<double>& b, std::string activation);

template<typename T>
void activationFunDerivative(std::vector<T>& a, std::vector<T>& b, std::string activation){
    //std::cout << "Passata activation function: " << activation << std::endl;
    for(int i = 0; i < a.size(); i++){
        b[i] = applyActivationFunctionDerivative(a[i], activation);
    }
}
template void activationFunDerivative<float>(std::vector<float>& a, std::vector<float>& b, std::string activation);
template void activationFunDerivative<double>(std::vector<double>& a, std::vector<double>& b, std::string activation);




//********************************************************************************************************************************************
//These functions given a m x n matrix return the transpose matrix, the first one return a new matrix, the second one modify the input matrix

template<typename T>
std::vector<T> transposeMatrix(const std::vector<T>& matrix, const int m, const int n){
    std::vector<T> transposed_matrix;
    transposed_matrix.resize(1);
    transposed_matrix.resize(m*n);
    for(int i = 0; i < n; i++){
        for(int j = 0; j < m; j++){        
            transposed_matrix[j+i*m] = matrix[i+j*n];
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
    for(int i=0; i<n; i++){
        for(int j=0; j<m; j++){
            transposed[j+i*m] = matrix[i+j*n];
        }
    }
}
template void transposeMatrix2<float>(const std::vector<float>& matrix, std::vector<float>& transposed,  const int m, const int n);
template void transposeMatrix2<double>(const std::vector<double>& matrix, std::vector<double>& transposed,  const int m, const int n);


//****************************************************************************************************************************************************
//This function defined in activation_functions.hpp compute the derivative of the mean square error (MSE)

template<typename T>
void mseDerivative( std::vector<T>& y, std::vector<T>& target, std::vector<T>& dE_dy){
    for(int i = 0; i < y.size(); i++){
        dE_dy[i] = y[i] - target[i];
    }
}
template void mseDerivative<float>(std::vector<float>& y, std::vector<float>& target, std::vector<float>& dE_dy);
template void mseDerivative<double>(std::vector<double>& y, std::vector<double>& target, std::vector<double>& dE_dy);

//****************************************************************************************************************************************************
/**
 * This function is used to redirect the evaluation of the derivative of the loss function to the correct one:
 *      1) MSE apply the derivative (Mean Square Error)
 **/

template<typename T>
void applyLossFunction(std::vector<T>& y,std::vector<T>& target, std::vector<T>& dE_dy, std::string& lossFunction){
    if(lossFunction == "MSE"){
        mseDerivative(y, target, dE_dy);
    }
    else{
        std::cout << "Error: loss function not recognized" << std::endl;
    }
}
template void applyLossFunction<float>(std::vector<float>& y, std::vector<float>& target, std::vector<float>& dE_dy, std::string& lossFunction);
template void applyLossFunction<double>(std::vector<double>& y, std::vector<double>& target, std::vector<double>& dE_dy, std::string& lossFunction);


//****************************************************************************************************************************************************
//This function defined in functions_utilities.hpp compute the mean square error (MSE)

template<typename T>
T mse(std::vector<T>& y, std::vector<T>& target){
    T result = 0;
    for(int i = 0; i < y.size(); i++){
        result += pow(y[i] - target[i], 2);
    }
    result = result / y.size();
    return result;
}
template float mse<float>(std::vector<float>& y, std::vector<float>& target);
template double mse<double>(std::vector<double>& y, std::vector<double>& target);


//****************************************************************************************************************************************************
/**
 * This function is used to redirect the loss function to the correct one:
 *      1) MSE (Mean Square Error)
 **/

template<typename T>
T evaluateLossFunction(std::vector<T>& y, std::vector<T>& target, std::string& lossFunction){
    T result;
    if(lossFunction == "MSE"){
        result = mse(y, target);
        return result;
    }
    else{
        std::cout << "Error: loss function not recognized" << std::endl;
        return result;
    }
}
template float evaluateLossFunction<float>(std::vector<float>& y, std::vector<float>& target, std::string& lossFunction);
template double evaluateLossFunction<double>(std::vector<double>& y, std::vector<double>& target, std::string& lossFunction);


/*
 * ***********************************************************************************************************************
 * *******************************  MODEL IMPLEMENTATIONS  ***************************************************************
 * ***********************************************************************************************************************
*/

//**************************************************************************************************************************
/**
 * This method print on the standatrd output all weights and bias matrix
 **/

template<typename T>
void Model<T>::printWeigts() const {
        for(int l=0; l <= layers.size(); l++){
            std::cout << "weights layer " << l+1 <<std::endl;
            for(int i = 0; i<weights_shape[l][0]; i++){
                for(int j =0 ; j<weights_shape[l][1]; j++){
                    std::cout << weights[l][j+i*weights_shape[l][1]] << " ";

                }
                std::cout << std::endl;
            }
            std::cout << std::endl;
            std::cout << "bias layer " << l+1 << std::endl;
            for(int i = 0; i<bias[l].size(); i++){
                std::cout << bias[l][i] << " ";
            }
            std::cout << std::endl;
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }
template void Model<float>::printWeigts() const;
template void Model<double>::printWeigts() const;

//**************************************************************************************************************************
/**
 * This Method produce a fancy print of the model in keras style, add information 
 * on the number of parameters and the activation function of each layer
*/

template<typename T>
void Model<T>::printModel() {
    std::cout << "Model name: " << model_name << std::endl;
        std::cout << "-------------------------------------------------------------" << std::endl;  
        std::cout << "Layers (type)" << std::setw(30) << "Output Shape" << std::setw(17) << "Params #" << std::endl;
        std::cout << "=============================================================" << std::endl;
        std::cout << std::endl;
        std::cout << "Input (Input Layer)" << "\033[38G" << model_input.getShapeInputData() << "\033[55G" << "0" <<  std::endl;
        std::cout << std::endl;
        int inValue = model_input.getShapeInputData();
        int totalParams = 0;
        for (int i = 0; i < layers.size(); ++i) {
            std::cout << layers[i].getName() << " (Dense)" <<  "\033[38G"  << layers[i].getNeurons() << "\033[55G" << layers[i].getNeurons()*(inValue+1) << std::endl;
            std::cout << std::endl;
            std::cout << layers[i].getActFun() << " (Hidden Layer)" << "\033[38G" << layers[i].getNeurons() << "\033[55G" << "0" <<  std::endl;
            std::cout << std::endl;
            totalParams += layers[i].getNeurons()*(inValue+1);
            inValue = layers[i].getNeurons();
        }
        //printLayers();
        //std::cout << std::endl;
        std::cout << "Output (Output layer)" << "\033[38G" << model_output.getShapeOutputData() << "\033[55G" << model_output.getShapeOutputData()*(inValue+1) <<  std::endl;
        totalParams += model_output.getShapeOutputData()*(inValue+1);
        std::cout << std::endl;
        std::cout << model_output.getAct_Fun() << " (Activation Function)" << "\033[38G" << model_output.getShapeOutputData() << "\033[55G" << "0" <<  std::endl;
        std::cout << std::endl;
        std::cout << "=============================================================" << std::endl;
        std::cout << std::endl;
        std::cout << "Total params: " << totalParams << " (" << totalParams*sizeof(T)/1024 << " KB)" << std::endl;
        //std::cout << std::endl;
        std::cout << "_____________________________________________________________" << std::endl;  
        std::cout << std::endl;

}

template void Model<float>::printModel();
template void Model<double>::printModel();



//**************************************************************************************************************************
//This function defined in the class Model build all matrix needed for the training and the prediction, starting from defined input, layers and output

template<typename T>
void Model<T>::buildModel(){
        std::cout << "Building model..." << std::endl;
        std::cout << "Model name: " << model_name << std::endl;
        std::cout << "Number of epochs: " << model_epochs << std::endl;
        std::cout << "Batch size: " << model_batch_size << std::endl;
        std::cout << "Learning rate: " << model_learning_rate << std::endl;
        std::cout << "Loss function: " << model_loss_fun << std::endl;
        std::cout << "Stop cryteria: " << model_stop_cryteria << std::endl;
        std::cout << std::endl;
        std::cout << "Input layer: " << std::endl;
        std::cout << "Number of input introduced in the network: " << model_input.getShapeInputData() << std::endl;

        std::cout << std::endl;
        std::cout << std::endl;
        printModel();
        
        //resizing the vectors
        weights.resize(layers.size()+1);
        bias.resize(layers.size()+1);
        weights_shape.resize(layers.size()+1);
        z.resize(layers.size()+1);
        h.resize(layers.size());
        dAct_z.resize(layers.size()+1);
        dE_dw.resize(layers.size()+1);
        dE_dx.resize(layers.size());
        dE_db.resize(layers.size()+1);
        y.resize(model_output.getShapeOutputData());

        //create dimensions for the first iteration
        int fillerdim = model_input.getShapeInputData();
        int check = 0;
        for(int block = 0; block < layers.size(); ++block){
            std::vector<T> filler(fillerdim * layers[block].getNeurons(), default_weight);
            std::vector<T> fillerBias(layers[block].getNeurons(), default_weight);
            weights_shape[block].resize(2);
            //changing the order of the dimensions will change the order of the weights in the matrix in relation
            //on how the input is passed to the network
            weights_shape[block][1] = layers[block].getNeurons();
            weights_shape[block][0] = fillerdim;
            weights[block].resize(fillerdim * layers[block].getNeurons());
            weights[block] = filler;
            bias[block].resize(layers[block].getNeurons());
            bias[block] = fillerBias;
            dE_dw[block].resize(fillerdim * layers[block].getNeurons());
            dAct_z[block].resize(layers[block].getNeurons());
            z[block].resize(layers[block].getNeurons());
            h[block].resize(layers[block].getNeurons());
            dE_dx[block].resize(layers[block].getNeurons());
            dE_db[block].resize(layers[block].getNeurons());
            dE_dx[block].resize(layers[block].getNeurons());
            //update the dimensions for the next iteration
            fillerdim = layers[block].getNeurons();
            check += 1;
        }
        std::vector<T> filler(layers[check-1].getNeurons() * model_output.getShapeOutputData(), default_weight);
        std::vector<T> fillerBias(model_output.getShapeOutputData(), default_weight);
        weights[check].resize(fillerdim * model_output.getShapeOutputData());
        weights[check] = filler;
        weights_shape[check].resize(2);
        weights_shape[check][1] = model_output.getShapeOutputData();
        weights_shape[check][0] = layers[check-1].getNeurons();
        bias[check].resize(model_output.getShapeOutputData());
        bias[check] = fillerBias;
        dE_dw[check].resize(fillerdim * model_output.getShapeOutputData());
        dAct_z[check].resize(model_output.getShapeOutputData());
        dE_db[check].resize(model_output.getShapeOutputData());
        z[check].resize(model_output.getShapeOutputData());
        y.resize(model_output.getShapeOutputData());
        initialiseVector(weights, weights_initialisation);
        initialiseVector(bias, weights_initialisation);



        std::cout << "Model built!" << std::endl;
        std::cout << std::endl;
    }
template void Model<float>::buildModel();
template void Model<double>::buildModel();

//**************************************************************************************************************************
//function used to print to weights.txt all the matrix of weight at a certain iteration

template<typename T>
void Model<T>::printAllWeightsToFile(){
    std::ofstream outputFile("weights.txt", std::ios::app);
    outputFile << "********************************NEW SET OF WEIGHTS**********************************************" << std::endl;
    outputFile << std::endl;
    for(int l=0; l <= layers.size(); l++){
            outputFile << "************* weights layer " << l+1 << " ****************" << std::endl;
            outputFile << std::endl;
            outputFile << "weigthts " << weights_shape[l][0] << " x " << weights_shape[l][1] << std::endl;
            for(int i = 0; i<weights_shape[l][0]; i++){
                for(int j =0 ; j<weights_shape[l][1]; j++){
                    outputFile << weights[l][j+i*weights_shape[l][1]] << " ";

                }
                outputFile << std::endl;
            }
                //add here all needed matrix and bias at the end
            outputFile << std::endl;
            outputFile << "bias layer " << l+1 << " size: " << bias[l].size() << std::endl;
            for(int i = 0; i<bias[l].size(); i++){
                outputFile << bias[l][i] << " ";
            }
            outputFile << std::endl;
            outputFile << std::endl;
            outputFile << "z input layer " << l+1 << " size: " << z[l].size() << std::endl;
                for(int i = 0; i<z[l].size(); i++){
                    outputFile << z[l][i] << " ";
                }
            outputFile << std::endl;
            outputFile << std::endl;
            if (l < layers.size()){
                outputFile << "h output layer " << l+1 << " size: " << h[l].size() << std::endl;
                for(int i = 0; i<h[l].size(); i++){
                    outputFile << h[l][i] << " ";
                }
                outputFile << std::endl;
                outputFile << std::endl;
                outputFile << "dE_dx layer " << l+1 << " size: " << dE_dx[l].size() << std::endl;
                for(int i = 0; i<dE_dx[l].size(); i++){
                    outputFile << dE_dx[l][i] << " ";
                }
                outputFile << std::endl;
                outputFile << std::endl;
            }
            
            outputFile << "dE_db layer " << l+1 << " size: " << dE_db[l].size() << std::endl;
            for(int i = 0; i<dE_db[l].size(); i++){
                outputFile << dE_db[l][i] << " ";
            }
            outputFile << std::endl;
            outputFile << std::endl;
            outputFile << "dE_dw layer " << l+1 << " size: " << weights_shape[l][0] << " x " << weights_shape[l][1] << std::endl;
            for(int i = 0; i<weights_shape[l][0]; i++){
                for(int j =0 ; j<weights_shape[l][1]; j++){
                    outputFile << dE_dw[l][j+i*weights_shape[l][1]] << " ";

                }
                outputFile << std::endl;
            }
            outputFile << std::endl;
    }
    outputFile << "y output layer " << layers.size()+1 << " size: " << y.size() << std::endl;
    for(int i = 0; i<y.size(); i++){
        outputFile << y[i] << " ";
    }
    outputFile << std::endl;
    outputFile.close();
}
template void Model<float>::printAllWeightsToFile();
template void Model<double>::printAllWeightsToFile();


//******************************************************************************************************************************************
//This function initialize the weights and the bias of the model, different model of initialization available 
//setting different values for the variable weights_model:

/*  1) DEFAULT CHOICE "Normal_Distribution"  Gaussian 0 mean and standard deviation 1 normal distribution
    2) "Uniform_distribution" Gaussian uniform distribution, 0 mean and 1 STD
    3) "He" optimized for ReLu Activadion Functions
    4) "Xavier" optimized for non ReLu Activetion Functions or Marvel's fans
    5) "debug" define weights with rows filled by the same value equal to row number +1 use for debug reasons */

template<typename T>
void Model<T>::initialiseVector(std::vector<std::vector<T>>& default_weights, const std::string& weights_model){
    if(weights_model == "Normal_Distribution"){
        //std::random_device rd;   //seed variabile
        //std::mt19937 gen(rd());
        std::mt19937 gen(44); //seed fisso
        std::normal_distribution<T> distribution(0.0, 0.1); // Distribuzione normale con media 0 e deviazione standard 0.1

        for (auto& row : default_weights) {
            for (T& weight : row) {
                weight = distribution(gen);
            }
        }   

    }
    else if(weights_model == "Uniform_Distribution"){
        //std::random_device rd;   //seed variabile
        //std::mt19937 gen(rd());
        std::mt19937 gen(44); //seed fisso
        std::uniform_real_distribution<T> distribution(0.0, 0.1); // Distribuzione normale con media 0 e deviazione standard 0.1

        for (auto& row : default_weights) {
            for (T& weight : row) {
                weight = distribution(gen);
            }
        }   

    }
    else if(weights_model == "Xavier"){
        //std::random_device rd;   //seed variabile
        //std::mt19937 gen(rd());
        std::mt19937 gen(44); //seed fisso
        std::normal_distribution<T> distribution(0.0, 1.0); // Distribuzione normale con media 0 e deviazione standard 1.0
        for (auto& row : default_weights) {
            for (T& weight : row) {
                weight = distribution(gen);
            }
        }   
        for(int i = 0; i < default_weights.size(); i++){
            for(int j = 0; j < default_weights[i].size(); j++){
                default_weights[i][j] = default_weights[i][j] * sqrt(1.0/weights_shape[i][0]); //weighted on size of the input
            }
        }
    }
    else if(weights_model == "He"){
        //std::random_device rd;   //seed variabile
        //std::mt19937 gen(rd());
        std::mt19937 gen(44); //seed fisso
        std::normal_distribution<T> distribution(0.0, 1.0); // Distribuzione normale con media 0 e deviazione standard 1.0
        for (auto& row : default_weights) {
            for (T& weight : row) {
                weight = distribution(gen);
            }
        }   
        for(int i = 0; i < default_weights.size(); i++){
            for(int j = 0; j < default_weights[i].size(); j++){
                default_weights[i][j] = default_weights[i][j] * sqrt(2.0/weights_shape[i][0]);  //weighted on size of the input
            }
        }
    }
    else if (weights_model == "debug"){
        for(int i = 0; i < default_weights.size(); i++){
            for(int j = 0; j < default_weights[i].size(); j++){
                default_weights[i][j] = i+1;
            }
        }
    }
    else {
        std::cout << "Error: weights model not recognized" << std::endl;
        return;
    }
    
}
template void Model<float>::initialiseVector(std::vector<std::vector<float>>& default_weights, const std::string& weights_model);
template void Model<double>::initialiseVector(std::vector<std::vector<double>>& default_weights, const std::string& weights_model);



//****************************************************************************************************************************************************
/**
 * These two function are used to build the matrix used in predict function:
 *     extendMatrix() add the vector of bias as last row in weight matrix, add 1 to the input vector to fix the dimensions
 *     reduceMatrix() remove the vector of bias as last row in weight matrix, remove 1 to the input vector to fix the dimensions
*/

template<typename T>
void Model<T>::extendMatrix(){
    //input.push_back(1);
    //weights[0].resize(weights_shape[0][0]+bias[0].size());
    weights[0].insert(weights[0].end(), bias[0].begin(), bias[0].end());
    for(int loop = 0; loop < layers.size(); loop++){
        h[loop].push_back(1);
        //weights[loop+1].resize(weights_shape[loop+1][0]+bias[loop+1].size());
        weights[loop+1].insert(weights[loop+1].end(), bias[loop+1].begin(), bias[loop+1].end());
    }
}

template void Model<float>::extendMatrix();
template void Model<double>::extendMatrix();

template<typename T>
void Model<T>::reduceMatrix(){
    //input.pop_back();
    //weights[0].erase(weights[0].end()-bias[0].size(), weights[0].end());
    weights[0].resize(weights_shape[0][0]*weights_shape[0][1]);
    for(int loop = 0; loop < layers.size(); loop++){
        h[loop].pop_back();
        //weights[loop+1].erase(weights[loop+1].end()-bias[loop+1].size(), weights[loop+1].end());
        weights[loop+1].resize(weights_shape[loop+1][0]*weights_shape[loop+1][1]);
    }
}

template void Model<float>::reduceMatrix();
template void Model<double>::reduceMatrix();



//****************************************************************************************************************************************************
/**
 * These two functions compute the forward probagation of the input along the network, producing as output the variable y
 * the first one take as input the input vector and the selection of the activation function to use, 
 * note that can be called only after the resizing of the weights, the second one take as input the input vector, 
 * the selection of the activation function to use and a flag, if the flag is any integer the function will extend the matrix of weights and bias,
 * usefull to be used in the main() function.
 * 
 * ***************IMPORTANT****************
 * when this function is called remember to reset to 0 the z vector, otherwise it will be summed to the following iterations !!!!!!!!!!!
*/

template<typename T> //thi version need to be called only after the resizing of the weights
void Model<T>::predict(std::vector<T>& input, int& selection){
    input.push_back(1);
    const auto t0_0 = std::chrono::high_resolution_clock::now();
    mul_funct(input, weights[0], z[0], 1, weights_shape[0][0]+1, weights_shape[0][1], matrix_mul_optimisation, cuda_block_size);   //weights[0], input, z, weights_shape[0][0], weights_shape[0][1], 1, matrix_mul_optimisation);
    const auto t0_1 = std::chrono::high_resolution_clock::now();
    int64_t dt_01 = std::chrono::duration_cast<std::chrono::microseconds>(t0_1 - t0_0).count();
    times[0] += dt_01;
    activationFun(z[0], h[0], layers[0].getActFun());
    
    for(int loop = 0; loop < layers.size(); loop++){
        const auto t1_0 = std::chrono::high_resolution_clock::now();
        mul_funct(h[loop], weights[loop+1], z[loop+1], 1, weights_shape[loop+1][0]+1, weights_shape[loop+1][1], matrix_mul_optimisation, cuda_block_size);
        const auto t1_1 = std::chrono::high_resolution_clock::now();
        int64_t dt_02 = std::chrono::duration_cast<std::chrono::microseconds>(t1_1 - t1_0).count();
        times[1+loop] += dt_02;
        if(loop < layers.size()-1){
            activationFun(z[loop+1], h[loop+1], layers[loop+1].getActFun());
        }
    }
    activationFun(z[layers.size()], y, model_output.getOutputAct_fun());
    input.pop_back();
}

template void Model<float>::predict(std::vector<float>& input, int& selection);
template void Model<double>::predict(std::vector<double>& input, int& selection);

template<typename T> //this version contains the extension and reduction of the matrix
void Model<T>::predict(std::vector<T>& input, int& selection, int flag){
    extendMatrix();
    input.push_back(1);
    const auto t0_0 = std::chrono::high_resolution_clock::now();
    mul_funct(input, weights[0], z[0], 1, weights_shape[0][0]+1, weights_shape[0][1], matrix_mul_optimisation, cuda_block_size);   //weights[0], input, z, weights_shape[0][0], weights_shape[0][1], 1, matrix_mul_optimisation);
    const auto t0_1 = std::chrono::high_resolution_clock::now();
    int64_t dt_01 = std::chrono::duration_cast<std::chrono::microseconds>(t0_1 - t0_0).count();
    times[0] += dt_01;
    activationFun(z[0], h[0], layers[0].getActFun());
    
    for(int loop = 0; loop < layers.size(); loop++){
        const auto t1_0 = std::chrono::high_resolution_clock::now();
        mul_funct(h[loop], weights[loop+1], z[loop+1], 1, weights_shape[loop+1][0]+1, weights_shape[loop+1][1], matrix_mul_optimisation, cuda_block_size);
        const auto t1_1 = std::chrono::high_resolution_clock::now();
        int64_t dt_02 = std::chrono::duration_cast<std::chrono::microseconds>(t1_1 - t1_0).count();
        times[1+loop] += dt_02;
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

//template<typename T> //this version is for cuda optimization
template<>
void Model<float>::predict(float *input, int selection){
    
    //input.push_back(1);
    //mul_funct(input, weights[0], z[0], 1, weights_shape[0][0]+1, weights_shape[0][1], matrix_mul_optimisation, cuda_block_size);   //weights[0], input, z, weights_shape[0][0], weights_shape[0][1], 1, matrix_mul_optimisation);
    const auto t0_0 = std::chrono::high_resolution_clock::now();
    cudaFunctionFOptimized(input, &cw[0], &cz[0], 1, weights_shape[0][0], weights_shape[0][1], cuda_block_size);
    const auto t0_1 = std::chrono::high_resolution_clock::now();
    int64_t dt_01 = std::chrono::duration_cast<std::chrono::microseconds>(t0_1 - t0_0).count();
    //std::cout << "time1: " << dt_01 <<  std::endl;
    times[0] += dt_01;
    for (int i = 0; i< bias[0].size(); i++){
        cz[i] = cz[i] + cbi[i];
    }
    //activationFun(z[0], h[0], layers[0].getActFun());
    for(int i = 0; i < z[0].size(); i++){
        ch[i] = (cz[i] > 0) ? cz[i] : 0;
    }
    
    for(int loop = 0; loop < layers.size(); loop++){
        //mul_funct(h[loop], weights[loop+1], z[loop+1], 1, weights_shape[loop+1][0]+1, weights_shape[loop+1][1], matrix_mul_optimisation, cuda_block_size);
        const auto t0_1 = std::chrono::high_resolution_clock::now();
        cudaFunctionFOptimized(&ch[add_ch[loop]], &cw[add_cw[loop+1]], &cz[add_cz[loop+1]], 1, weights_shape[loop+1][0], weights_shape[loop+1][1], cuda_block_size);
        const auto t1_1 = std::chrono::high_resolution_clock::now();
        int64_t dt_02 = std::chrono::duration_cast<std::chrono::microseconds>(t1_1 - t0_1).count();
        
        times[1+loop] += dt_02;
        for (int i = 0; i< bias[loop+1].size(); i++){
            cz[add_cz[loop+1]+i] = cz[add_cz[loop+1]+i] + cbi[add_cbi[loop+1]+i];
        }
        if(loop < layers.size()-1){
            //activationFun(z[loop+1], h[loop+1], layers[loop+1].getActFun());
            for(int i = 0; i < z[loop+1].size(); i++){
                ch[add_ch[loop+1]+i] = (cz[add_cz[loop+1]+i] > 0) ? cz[add_cz[loop+1]+i] : 0;
            }
        }
    }
    //activationFun(z[layers.size()], y, model_output.getOutputAct_fun());
    for(int i = 0; i < y.size(); i++){
        cy[i] = 1 / (1 + std::exp(-cz[add_cz[layers.size()]+i]));
    }
    //input.pop_back();
    //std::cout << std::endl << "sono arrivato qui 4" << std::endl << std::flush;
}
//template void Model<float>::predict(float *input, int selection);
//template void Model<double>::predict(double *input, int selection);

template<>
void Model<double>::predict(double *input, int selection){
    return;
}

//****************************************************************************************************************************************************
//This function defined in Model.hpp compute the backpropagation of the model using the chain rule and Gradient Descent

template<typename T>
void Model<T>::backPropagation(std::vector<T>& input, std::vector<T>& dE_dy, int& selection){
    int one=1;
    std::vector<T> temp;
    activationFunDerivative(z[layers.size()], dAct_z[layers.size()], model_output.getOutputAct_fun());
    dE_db[layers.size()] = mul(dE_dy, dAct_z[layers.size()]);
    temp = transposeMatrix(h[layers.size()-1], one, h[layers.size()-1].size());
    const auto t0_0 = std::chrono::high_resolution_clock::now();
    mul_funct(temp , dE_db[layers.size()],dE_dw[layers.size()], h[layers.size()-1].size(), one, dE_db[layers.size()].size(), matrix_mul_optimisation, cuda_block_size);
    const auto t0_1 = std::chrono::high_resolution_clock::now();
    int64_t dt_01 = std::chrono::duration_cast<std::chrono::microseconds>(t0_1 - t0_0).count();
    times[1+layers.size()] += dt_01;
    transposeMatrix2(weights[layers.size()], temp, weights_shape[layers.size()][0], weights_shape[layers.size()][1]);
    const auto t1_0 = std::chrono::high_resolution_clock::now();
    mul_funct(dE_db[layers.size()], temp, dE_dx[layers.size()-1], one,  dE_db[layers.size()].size(), weights_shape[layers.size()][0], matrix_mul_optimisation, cuda_block_size);
    const auto t1_1 = std::chrono::high_resolution_clock::now();
    int64_t dt_02 = std::chrono::duration_cast<std::chrono::microseconds>(t1_1 - t1_0).count();
    times[1+layers.size()+1] += dt_02;
    for (int i=layers.size()-1; i > 0; i--){
        activationFunDerivative(z[i], dAct_z[i], layers[i].getActFun());
        dE_db[i] = mul(dE_dx[i], dAct_z[i]);
        temp = transposeMatrix(h[i-1], one, h[i-1].size());
        const auto t2_0 = std::chrono::high_resolution_clock::now();
        mul_funct(temp, dE_db[i], dE_dw[i], h[i-1].size(), one, dE_db[i].size(), matrix_mul_optimisation, cuda_block_size);
        const auto t2_1 = std::chrono::high_resolution_clock::now();
        int64_t dt_03 = std::chrono::duration_cast<std::chrono::microseconds>(t2_1 - t2_0).count();
        times[1+layers.size()+1+1+i] += dt_03;
        transposeMatrix2(weights[i], temp, weights_shape[i][0], weights_shape[i][1]);
        const auto t3_0 = std::chrono::high_resolution_clock::now();
        mul_funct(dE_db[i], temp, dE_dx[i-1], one,  dE_db[i].size(), weights_shape[i][0], matrix_mul_optimisation, cuda_block_size);
        const auto t3_1 = std::chrono::high_resolution_clock::now();
        int64_t dt_04 = std::chrono::duration_cast<std::chrono::microseconds>(t3_1 - t3_0).count();
        times[1+layers.size()+1+1+i+layers.size()-1] += dt_04;
    }
    activationFunDerivative(z[0], dAct_z[0], layers[0].getActFun());
    dE_db[0] = mul(dE_dx[0], dAct_z[0]);
    temp = transposeMatrix(input, one, input.size());
    const auto t4_0 = std::chrono::high_resolution_clock::now();
    mul_funct(temp, dE_db[0], dE_dw[0], input.size(), one, dE_db[0].size(), matrix_mul_optimisation, cuda_block_size);
    const auto t4_1 = std::chrono::high_resolution_clock::now();
    int64_t dt_05 = std::chrono::duration_cast<std::chrono::microseconds>(t4_1 - t4_0).count();
    times[4 + 1*layers.size() + 2*(layers.size()-1)-1] += dt_05;
    
}
template void Model<float>::backPropagation(std::vector<float>& input, std::vector<float>& dE_dy, int& selection);
template void Model<double>::backPropagation(std::vector<double>& input, std::vector<double>& dE_dy, int& selection);


//cuda version

//template<typename T>
template<>
void Model<float>::backPropagation(float *input, float *cdE_dy, int selection){
    int one=1;
    int m,n;
    std::vector<float> temp;
    //activationFunDerivative(z[layers.size()], dAct_z[layers.size()], model_output.getOutputAct_fun());
    for(int i = 0; i < z[layers.size()].size(); i++){
        cdAct_z[add_cdAct_z[layers.size()]+i] = 1 / (1 + std::exp(-cdAct_z[add_cdAct_z[layers.size()]+i]));
        cdAct_z[add_cdAct_z[layers.size()]+i] = cdAct_z[add_cdAct_z[layers.size()]+i] * (1 - cdAct_z[add_cdAct_z[layers.size()]+i]);
    }
    //dE_db[layers.size()] = mul(dE_dy, dAct_z[layers.size()]);
    for (int i = 0; i< bias[layers.size()].size(); i++){
        cdE_db[add_cdE_db[layers.size()]+i] = cdE_dy[i] * cdAct_z[add_cdAct_z[layers.size()]+i];
    }
    //temp = transposeMatrix(h[layers.size()-1], one, h[layers.size()-1].size());
    //mul_funct(temp , dE_db[layers.size()],dE_dw[layers.size()], h[layers.size()-1].size(), one, dE_db[layers.size()].size(), matrix_mul_optimisation, cuda_block_size);
    const auto t0_0 = std::chrono::high_resolution_clock::now();
    cudaFunctionFOptimized(&ch[add_ch[layers.size()-1]], &cdE_db[add_cdE_db[layers.size()]], &cdE_dw[add_cdE_dw[layers.size()]], h[layers.size()-1].size(), one, dE_db[layers.size()].size(), cuda_block_size);
    const auto t0_1 = std::chrono::high_resolution_clock::now();
    int64_t dt_01 = std::chrono::duration_cast<std::chrono::microseconds>(t0_1 - t0_0).count();
    times[1+layers.size()] += dt_01;
    //transposeMatrix2(weights[layers.size()], temp, weights_shape[layers.size()][0], weights_shape[layers.size()][1]);
    m=weights_shape[layers.size()][0];
    n=weights_shape[layers.size()][1];
    for(int i=0; i<n; i++){
        for(int j=0; j<m; j++){
            trsp_cw[add_trsp_cw[layers.size()]+j+i*m] = cw[add_cw[layers.size()]+i+j*n];
        }
    }
    //mul_funct(dE_db[layers.size()], temp, dE_dx[layers.size()-1], one,  dE_db[layers.size()].size(), weights_shape[layers.size()][0], matrix_mul_optimisation, cuda_block_size);
    const auto t1_0 = std::chrono::high_resolution_clock::now();
    cudaFunctionFOptimized(&cdE_db[add_cdE_db[layers.size()]], &trsp_cw[add_trsp_cw[layers.size()]], &cdE_dx[add_cdE_dx[layers.size()-1]], one,  dE_db[layers.size()].size(), weights_shape[layers.size()][0], cuda_block_size);
    const auto t1_1 = std::chrono::high_resolution_clock::now();
    int64_t dt_02 = std::chrono::duration_cast<std::chrono::microseconds>(t1_1 - t1_0).count();
    times[1+layers.size()+1] += dt_02;
    for (int i=layers.size()-1; i > 0; i--){
        //activationFunDerivative(z[i], dAct_z[i], layers[i].getActFun());
        for(int j=0;j<dAct_z[i].size();j++){
            cdAct_z[add_cdAct_z[i]+j] = (cz[add_cz[i]+j] > 0) ? 1 : 0;
        }
        //dE_db[i] = mul(dE_dx[i], dAct_z[i]);
        for(int j=0;j<dE_dx[i].size();j++){
            cdE_db[add_cdE_db[i]+j] = cdE_dx[add_cdE_dx[i]+j] * cdAct_z[add_cdAct_z[i]+j];
        }
        //temp = transposeMatrix(h[i-1], one, h[i-1].size());
        //mul_funct(temp, dE_db[i], dE_dw[i], h[i-1].size(), one, dE_db[i].size(), matrix_mul_optimisation, cuda_block_size);
        const auto t2_0 = std::chrono::high_resolution_clock::now();
        cudaFunctionFOptimized(&ch[add_ch[i-1]], &cdE_db[add_cdE_db[i]], &cdE_dw[add_cdE_dw[i]], h[i-1].size(), one, dE_db[i].size(), cuda_block_size);
        const auto t2_1 = std::chrono::high_resolution_clock::now();
        int64_t dt_03 = std::chrono::duration_cast<std::chrono::microseconds>(t2_1 - t2_0).count();
        times[1+layers.size()+1+1+i] += dt_03;
        //transposeMatrix2(weights[i], temp, weights_shape[i][0], weights_shape[i][1]);
        m=weights_shape[i][0];
        n=weights_shape[i][1];
        for(int ii=0; ii<n; ii++){
            for(int j=0; j<m; j++){
                trsp_cw[add_trsp_cw[i]+j+ii*m] = cw[add_cw[i]+ii+j*n];
            }
        }
        //mul_funct(dE_db[i], temp, dE_dx[i-1], one,  dE_db[i].size(), weights_shape[i][0], matrix_mul_optimisation, cuda_block_size);
        const auto t3_0 = std::chrono::high_resolution_clock::now();
        cudaFunctionFOptimized(&cdE_db[add_cdE_db[i]], &trsp_cw[add_trsp_cw[i]], &cdE_dx[add_cdE_dx[i-1]], one,  dE_db[i].size(), weights_shape[i][0], cuda_block_size);
        const auto t3_1 = std::chrono::high_resolution_clock::now();
        int64_t dt_04 = std::chrono::duration_cast<std::chrono::microseconds>(t3_1 - t3_0).count();
        times[1+layers.size()+1+1+i+layers.size()-1] += dt_04;
    }
    //activationFunDerivative(z[0], dAct_z[0], layers[0].getActFun());
    for(int j=0;j<dAct_z[0].size();j++){
        cdAct_z[add_cdAct_z[0]+j] = (cz[add_cz[0]+j] > 0) ? 1 : 0;
    }
    dE_db[0] = mul(dE_dx[0], dAct_z[0]);
    for(int j=0;j<dE_dx[0].size();j++){
        cdE_db[j] = cdE_dx[+j] * cdAct_z[j];
    }
    //temp = transposeMatrix(input, one, input.size());
    //mul_funct(temp, dE_db[0], dE_dw[0], input.size(), one, dE_db[0].size(), matrix_mul_optimisation, cuda_block_size);
    const auto t4_0 = std::chrono::high_resolution_clock::now();
    cudaFunctionFOptimized(input, &cdE_db[0], &cdE_dw[0], model_input.getTrain()[0].size(), one, dE_db[0].size(), cuda_block_size);
    const auto t4_1 = std::chrono::high_resolution_clock::now();
    int64_t dt_05 = std::chrono::duration_cast<std::chrono::microseconds>(t4_1 - t4_0).count();
    times[4 + 1*layers.size() + 2*(layers.size()-1)-1] += dt_05;
    //std::cout << std::endl << "sono arrivato qui 5" << std::endl << std::flush;
}
//template void Model<float>::backPropagation(float *input, float *dE_dy, int selection);
//template void Model<double>::backPropagation(double *input, double *dE_dy, int selection);

template<>
void Model<double>::backPropagation(double *input, double *cdE_dy, int selection){
    return;
}


//****************************************************************************************************************************************************
/**
 * This function defined in Model.hpp take as input the chosen matrix multiplication algorithm chosen with "selection"
 * and train the parameters of the model using predict and backpropagation function
 **/

template<typename T>
void Model<T>::train(int& selection){
    int time_seize = 4 + 1*layers.size() + 2*(layers.size()-1);
    times.resize(time_seize);
    for(int i = 0; i < time_seize; i++){
        times[i] = 0;
    }
    std::vector<int64_t> times2;
    times2.resize(4);
    for(int i = 0; i < 4; i++){
        times2[i] = 0;
    }
    std::ofstream profileFile("Time_profile_fake.csv", std::ios::app);
    std::ofstream outputFile("Train_Output.txt");
    std::ofstream accuracyCSV("Accuracy.csv");
    std::ofstream lossCSV("Loss.csv");
    accuracyCSV << "epoch, train_accuracy, validation_accuracy" << std::endl;
    lossCSV << "epoch, batch, loss" << std::endl;
    std::vector<std::vector<T>> tempWeights = createTempWeightMAtrix(weights);
    std::vector<std::vector<T>> tempBias = createTempBiasMAtrix(bias);
    int batch = model_input.getTrain().size() / model_batch_size;
    int count=0, operations = 0, correct = 0;
    //float maxElement_train, max_element_target;
    int index_max_element_train, index_max_element_target; 
    float train_accuracy, validation_accuracy;
    std::vector<T> y_acc;
    std::vector<T> temp, temp2;
    dE_dy.resize(model_output.getShapeOutputData());
    temp.resize(model_input.getShapeInputData());
    y_acc.resize(model_output.getShapeOutputData());
    outputFile << "batch: " << batch << std::endl;
    outputFile << "train size: " << model_input.getTrain().size() << std::endl;
    std::cout << "Train started !  (details and results available in Train_Output.txt file)" << std::endl;
    std::cout << std::endl;
    //std::cout << "Progress: " ;
    const auto t0_0 = std::chrono::high_resolution_clock::now();
    int total_opp = 0;
    for(int epoch = 0; epoch < model_epochs; epoch++){
        outputFile << "epoch: " << std::setw(4) << epoch << " batch loop: " << batch << "/(";
        operations = 0;
        correct = 0;
        const auto t0 = std::chrono::high_resolution_clock::now();
        T loss = 0;
        for(int batch_loop = 0; batch_loop < batch+1; batch_loop++){//considera di aggiungere +1 per l'avanzo delle rimaneti singole batch
            outputFile << batch_loop;
            int percentage;
            percentage = ((epoch*batch)+batch_loop)*100/(model_epochs*batch);
            count = 0;
            //loss = 0;      \\uncomment if you need to evaluate the loss inside each batch, remember to uncomment also loss = loss/count at the end of the loop    
            for(int i = 0; i < model_batch_size; i++){
                if (operations < model_input.getTrain().size()){
                    const auto tt0 = std::chrono::high_resolution_clock::now();
                    total_opp++;
                    temp = model_input.getTrain()[batch_loop*model_batch_size+i];
                    extendMatrix();         //before predict call and for every predict in batch
                    const auto tt1 = std::chrono::high_resolution_clock::now();
                    predict(temp, selection);
                    const auto tt2 = std::chrono::high_resolution_clock::now();
                    reduceMatrix();          //after predict call and for every predict in batch
                    //cu_input = model_input.getTrain()[batch_loop*model_batch_size+i];
                    //predict(cu_input, selection, cu_weights, cu_bias, cu_z, cu_h);
                    applyLossFunction(y, model_output.getOutputTrain()[batch_loop*model_batch_size+i], dE_dy, model_loss_fun);
                    const auto tt3 = std::chrono::high_resolution_clock::now();
                    backPropagation(temp, dE_dy, selection);
                    const auto tt4 = std::chrono::high_resolution_clock::now();
                    incrementweightsBias(dE_dw, dE_db, tempWeights, tempBias);
                    loss += evaluateLossFunction(y, model_output.getOutputTrain()[batch_loop*model_batch_size+i], model_loss_fun);
                    resetVector(dE_dw);
                    resetVector(dE_dx);
                    resetVector(z);
                    index_max_element_target = 0;
                    float temp_1 = model_output.getOutputTrain()[batch_loop*model_batch_size+i][0];
                    for(int q =1; q<model_output.getOutputTrain()[batch_loop*model_batch_size+i].size(); q++){
                        if(model_output.getOutputTrain()[batch_loop*model_batch_size+i][q] > temp_1 ){
                            index_max_element_target = q;
                            temp_1 = model_output.getOutputTrain()[batch_loop*model_batch_size+i][q];
                        }
                    }
                    index_max_element_train = 0;
                    float temp_2 = y[0];
                    for(int q =1; q<y.size(); q++){
                        if(y[q] > temp_2 ){
                            index_max_element_train = q;
                            temp_2 = y[q];
                        }
                    }
                    if(index_max_element_target == index_max_element_train){
                        correct++;
                    }
                    operations++;
                    count++;
                    const auto tt5 = std::chrono::high_resolution_clock::now();
                    int64_t tt_00 = std::chrono::duration_cast<std::chrono::microseconds>(tt5 - tt0).count();
                    int64_t tt_01 = std::chrono::duration_cast<std::chrono::microseconds>(tt2 - tt1).count();
                    int64_t tt_02 = std::chrono::duration_cast<std::chrono::microseconds>(tt4 - tt3).count();
                    int64_t tt_03 = std::chrono::duration_cast<std::chrono::microseconds>(tt5 - tt4).count();
                    times2[0] += tt_00;
                    times2[1] += tt_01;
                    times2[2] += tt_02;
                    times2[3] += tt_03;
                }
            }
            updateWeightsBias(weights, tempWeights, bias, tempBias, count, model_learning_rate);
            resetVector(tempWeights);
            resetVector(tempBias);
            std::cout << "\r" << "progress: " << percentage << "%" << std::flush;
        }
        const auto t1 = std::chrono::high_resolution_clock::now();
        int64_t dt_01 = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
        train_accuracy = (float)correct/operations;
        outputFile << ") train Accuracy: " << std::setw(9) << train_accuracy ;

        loss = loss / operations;       // uncomment if you need to evaluate the loss at the end of each epoch
        lossCSV << epoch << "," << batch << "," << loss << std::endl;
        
        //evaluating accuracy on validation set
        std::vector<T> temp_validation;
        int correct_validation = 0;
        int operations_validation = 0;
        for(int i = 0; i < model_input.getValidation().size(); i++){
            temp_validation = model_input.getValidation()[i];
            extendMatrix(); //before predict call and for every predict in batch
            predict(temp_validation, selection);
            reduceMatrix(); //after predict call and for every predict in batch
            resetVector(z);
            index_max_element_target = 0;
            float temp_1 = model_output.getOutputValidation()[i][0];
            for(int q =1; q<model_output.getOutputValidation()[i].size(); q++){
                if(model_output.getOutputValidation()[i][q] > temp_1 ){
                    index_max_element_target = q;
                    temp_1 = model_output.getOutputValidation()[i][q];
                }
            }
            index_max_element_train = 0;
            float temp_2 = y[0];
            for(int q =1; q<y.size(); q++){
                if(y[q] > temp_2 ){
                    index_max_element_train = q;
                    temp_2 = y[q];
                }
            }
            if(index_max_element_target == index_max_element_train){
                correct_validation++;
            }
            operations_validation++;
        }
        validation_accuracy = (float)correct_validation/operations_validation;
        outputFile << "  validation Accuracy: " << std::setw(9) << validation_accuracy ;
        outputFile << "  time: " << dt_01 << " ms" << std::endl << std::flush;
        accuracyCSV << epoch << "," << train_accuracy << "," << validation_accuracy << std::endl;
        //accuracyCSV << std::endl;
    }
    outputFile << std::endl;
    outputFile << "operations: " << operations << std::endl;
    std::cout << std::endl;
    //profileFile << layers[0].getNeurons() << "," << layers[1].getNeurons() << "," << layers[2].getNeurons()  << "," << cuda_block_size << ",";
    for(int i = 0; i < times.size(); i++){
    //    std::cout << "time mul " << i << ": " << (float)times[i]/total_opp << " mics, total: " <<times[i]<< std::endl;
     //   profileFile << (float)times[i]/total_opp << ",";
    }
    std::cout << std::endl;
    for(int i = 0; i < times2.size(); i++){
      //  std::cout << "time2 " << i << ": " << (float)times2[i]/total_opp << " mics, total: " << times2[i] << std::endl;
      //  profileFile << (float)times2[i]/total_opp << ",";
    }
    profileFile << "1" << std::endl;
    std::cout << "Total mul: " << total_opp << std::endl;

    //evaluating accuracy on test set
    std::vector<T> temp_test;
    int correct_test = 0;
    int operations_test = 0;
    for(int i = 0; i < model_input.getTest().size(); i++){
        temp_test = model_input.getTest()[i];
        extendMatrix(); //before predict call and for every predict in batch
        predict(temp_test, selection);
        reduceMatrix(); //after predict call and for every predict in batch
        resetVector(z);
        
        index_max_element_target = 0;
        float temp_1 = model_output.getOutputTest()[i][0];
        for(int q =1; q<model_output.getOutputTest()[i].size(); q++){
            if(model_output.getOutputTest()[i][q] > temp_1 ){
                index_max_element_target = q;
                temp_1 = model_output.getOutputTest()[i][q];
            }
        }
        index_max_element_train = 0;
        float temp_2 = y[0];
        for(int q =1; q<y.size(); q++){
            if(y[q] > temp_2 ){
                index_max_element_train = q;
                temp_2 = y[q];
            }
        }
        if(index_max_element_target == index_max_element_train){
            correct_test++;
        }
        operations_test++;
    }
    float test_accuracy = (float)correct_test/operations_test;
    outputFile << std::endl;
    outputFile << "Final Accuracy on the TestSet: " << test_accuracy << std::endl;
    outputFile << std::endl;
    std::cout << std::endl;
    std::cout << "Final Accuracy on the TestSet: " << test_accuracy << std::endl;
    outputFile << std::endl;
    const auto t1_0 = std::chrono::high_resolution_clock::now();
    int64_t dt_00 = std::chrono::duration_cast<std::chrono::milliseconds>(t1_0 - t0_0).count();
    std::cout << std::endl;
    std::cout << std::endl;
    std::cout << "Train and evaluation on Test-Set successfully completed in " << (float)dt_00/1000 << " sec !" << std::endl;
    std::cout << std::endl;
    outputFile.close(); 
    accuracyCSV.close();
    lossCSV.close(); 
    profileFile.close();
    
            
}

template void Model<float>::train(int& selection);
template void Model<double>::train(int& selection);

//Cuda version

template<typename T>
void Model<T>::train(int &selection, int flag){
    int time_seize = 4 + 1*layers.size() + 2*(layers.size()-1);
    times.resize(time_seize);
    for(int i = 0; i < time_seize; i++){
        times[i] = 0;
    }
    std::vector<int64_t> times2;
    times2.resize(4);
    for(int i = 0; i < 4; i++){
        times2[i] = 0;
    }
    std::ofstream profileFile("Time_profile_fake.csv", std::ios::app);
    std::ofstream outputFile("cuda_Train_Output.txt");
    std::ofstream accuracyCSV("cuda_Accuracy.csv");
    std::ofstream lossCSV("cuda_Loss.csv");
    accuracyCSV << "epoch, train_accuracy, validation_accuracy" << std::endl;
    lossCSV << "epoch, batch, loss" << std::endl;
    //std::vector<std::vector<T>> tempWeights = createTempWeightMAtrix(weights);
    //std::vector<std::vector<T>> tempBias = createTempBiasMAtrix(bias);
    int batch = model_input.getTrain().size() / model_batch_size;
    int count=0, operations = 0, correct = 0;
    //float maxElement_train, max_element_target;
    int index_max_element_train, index_max_element_target; 
    //float *train_accuracy;
    //cudaMallocManaged((void **) &train_accuracy, sizeof(float));
    //*train_accuracy = 0.0;
    float validation_accuracy, train_accuracy;
    std::vector<T> y_acc;
    //std::vector<T> temp, temp2;
    //dE_dy.resize(model_output.getShapeOutputData());
    //temp.resize(model_input.getShapeInputData());
    //y_acc.resize(model_output.getShapeOutputData());
    outputFile << "batch: " << batch << std::endl;
    outputFile << "train size: " << model_input.getTrain().size() << std::endl;
    std::cout << "Train started !  (details and results available in Train_Output.txt file)" << std::endl;
    std::cout << std::endl;
    //std::cout << "Progress: " ;
    const auto t0_0 = std::chrono::high_resolution_clock::now();
    //std::cout << std::endl << "sono arrivato qui 1" << std::endl <<std::flush;
    T *loss;
    cudaMallocManaged((void **) &loss, sizeof(float));
    T *target;
    cudaMallocManaged((void **) &target, sizeof(float)*model_output.getOutputTrain()[0].size());
    T *temp_2;
    cudaMallocManaged((void **) &temp_2, sizeof(float)*model_input.getTrain()[0].size());
    *temp_2 = 0.0;
    //std::cout << std::endl << "sono arrivato qui 2" << std::endl << std::flush;
    int total_opp = 0;
    for(int epoch = 0; epoch < model_epochs; epoch++){
        outputFile << "epoch: " << std::setw(4) << epoch << " batch loop: " << batch << "/(";
        operations = 0;
        correct = 0;
        const auto t0 = std::chrono::high_resolution_clock::now();
        //T loss = 0;
        *loss = 0.0;
        for(int batch_loop = 0; batch_loop < batch+1; batch_loop++){//considera di aggiungere +1 per l'avanzo delle rimaneti singole batch
            outputFile << batch_loop;
            int percentage;
            percentage = ((epoch*batch)+batch_loop)*100/(model_epochs*batch);
            count = 0;
            //loss = 0;      \\uncomment if you need to evaluate the loss inside each batch, remember to uncomment also loss = loss/count at the end of the loop    
            for(int i = 0; i < model_batch_size; i++){
                if (operations < model_input.getTrain().size()){
                    const auto tt0 = std::chrono::high_resolution_clock::now();
                    total_opp++;
                    //temp = model_input.getTrain()[batch_loop*model_batch_size+i];
                    cudaMemcpy(in, model_input.getTrain()[batch_loop*model_batch_size+i].data(), model_input.getTrain()[batch_loop*model_batch_size+i].size()*sizeof(T), cudaMemcpyHostToDevice);
                    cudaMemcpy(target, model_output.getOutputTrain()[batch_loop*model_batch_size+i].data(), model_output.getOutputTrain()[batch_loop*model_batch_size+i].size()*sizeof(T), cudaMemcpyHostToDevice);
                    //extendMatrix();         //before predict call and for every predict in batch
                    //std::cout << std::endl << "sono arrivato qui 3" << std::endl << std::flush;
                    const auto tt1 = std::chrono::high_resolution_clock::now();
                    predict(in, selection);
                    const auto tt2 = std::chrono::high_resolution_clock::now();
                    //reduceMatrix();          //after predict call and for every predict in batch
                    //cu_input = model_input.getTrain()[batch_loop*model_batch_size+i];
                    //predict(cu_input, selection, cu_weights, cu_bias, cu_z, cu_h);
                    //applyLossFunction(y, model_output.getOutputTrain()[batch_loop*model_batch_size+i], dE_dy, model_loss_fun);
                    for(int ii = 0; ii < y.size(); ii++){
                       cdE_dy[ii] = cy[ii] - target[ii];
                    }
                    const auto tt3 = std::chrono::high_resolution_clock::now();
                    backPropagation(in, cdE_dy, selection);
                    const auto tt4 = std::chrono::high_resolution_clock::now();
                    for(int ii = 0; ii < weights.size(); ii++){
                        for(int jj = 0; jj < weights[ii].size(); jj++){
                            tmp_cw[add_tmp_cw[ii]+jj] += cdE_dw[add_cdE_dw[ii]+jj];
                        }
                    }
                    //std::cout << std::endl << "sono arrivato qui 6" << std::endl << std::flush;
                    //incrementweightsBias(dE_dw, dE_db, tempWeights, tempBias);
                    //loss += evaluateLossFunction(y, model_output.getOutputTrain()[batch_loop*model_batch_size+i], model_loss_fun);
                    for(int ii = 0; ii< y.size(); ii++){
                        *loss = *loss + (y[ii] - target[ii])*(y[ii] - target[ii]);
                    }
                    //std::cout << std::endl << "sono arrivato qui 7" << std::endl << std::flush;
                    //resetVector(dE_dw);
                    for(int ii = 0; ii < weights.size(); ii++){
                        for(int jj = 0; jj < weights[ii].size(); jj++){
                            cdE_dw[add_cdE_dw[ii]+jj] = 0.0;
                        }
                    }
                    //std::cout << std::endl << "sono arrivato qui 8" << std::endl << std::flush;
                    //resetVector(dE_dx);
                    for (int ii = 0; ii < dE_dx.size(); ii++){
                        for(int jj = 0; jj < dE_dx[ii].size(); jj++){
                            cdE_dx[add_cdE_dx[ii]+jj] = 0.0;
                        }
                    }
                    //std::cout << std::endl << "sono arrivato qui 9" << std::endl << std::flush;
                    //resetVector(z);
                    int erase_index = 0;
                    for (int ii = 0; ii < z.size(); ii++){
                        for(int jj = 0; jj < z[ii].size(); jj++){
                            cz[erase_index] = 0.0;
                            erase_index++;
                        }
                    }
                    //std::cout << std::endl << "sono arrivato qui 6" << std::endl << std::flush;
                    index_max_element_target = 0;
                    float temp_1 = model_output.getOutputTrain()[batch_loop*model_batch_size+i][0];
                    for(int q =1; q<model_output.getOutputTrain()[batch_loop*model_batch_size+i].size(); q++){
                        if(model_output.getOutputTrain()[batch_loop*model_batch_size+i][q] > temp_1 ){
                            index_max_element_target = q;
                            temp_1 = model_output.getOutputTrain()[batch_loop*model_batch_size+i][q];
                        }
                    }
                    index_max_element_train = 0;
                    *temp_2 = y[0];
                    for(int q =1; q<y.size(); q++){
                        if(y[q] > *temp_2 ){
                            index_max_element_train = q;
                            *temp_2 = y[q];
                        }
                    }
                    if(index_max_element_target == index_max_element_train){
                        correct++;
                    }
                    operations++;
                    count++;
                    const auto tt5 = std::chrono::high_resolution_clock::now();
                    int64_t tt_00 = std::chrono::duration_cast<std::chrono::microseconds>(tt5 - tt0).count();
                    int64_t tt_01 = std::chrono::duration_cast<std::chrono::microseconds>(tt2 - tt1).count();
                    int64_t tt_02 = std::chrono::duration_cast<std::chrono::microseconds>(tt4 - tt3).count();
                    int64_t tt_03 = std::chrono::duration_cast<std::chrono::microseconds>(tt5 - tt4).count();
                    times2[0] += tt_00;
                    times2[1] += tt_01;
                    times2[2] += tt_02;
                    times2[3] += tt_03;
                }
                //std::cout << std::endl << "sono arrivato qui 6" << std::endl << std::flush;
            }
            //updateWeightsBias(weights, tempWeights, bias, tempBias, count, model_learning_rate);
            for(int ii=0; ii<weights.size(); ii++){
                for(int jj=0; jj<weights[ii].size(); jj++){
                    cw[add_cw[ii]+jj] = cw[add_cw[ii]+jj] - model_learning_rate * tmp_cw[add_cw[ii]+jj] / count;
                }
                for(int jj=0; jj<bias[ii].size(); jj++){
                    cbi[add_cw[ii]+jj] = cbi[add_cw[ii]+jj] - model_learning_rate * tmp_cbi[add_cw[ii]+jj] / count;
                }
            }
            //resetVector(tempWeights);
            for(int ii=0; ii<weights.size(); ii++){
                for(int jj=0; jj<weights[ii].size(); jj++){
                    tmp_cw[add_cw[ii]+jj] = 0.0;
                }
                for(int jj=0; jj<bias[ii].size(); jj++){
                    tmp_cbi[add_cw[ii]+jj] = 0.0;
                }
            }
            //resetVector(tempBias);
            std::cout << "\r" << "progress: " << percentage << "%" << std::flush;
        }
        const auto t1 = std::chrono::high_resolution_clock::now();
        int64_t dt_01 = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
        train_accuracy = (float)correct/operations;
        outputFile << ") train Accuracy: " << std::setw(9) << train_accuracy ;

        *loss = *loss / operations;       // uncomment if you need to evaluate the loss at the end of each epoch
        lossCSV << epoch << "," << batch << "," << loss << std::endl;
        
        //evaluating accuracy on validation set
        std::vector<T> temp_validation;
        int correct_validation = 0;
        int operations_validation = 0;
        for (int i=0; i< weights.size();i++){
            cudaMemcpy(weights[i].data(), &cw[add_cw[i]], weights[i].size()*sizeof(T), cudaMemcpyDeviceToHost);
            cudaMemcpy(bias[i].data(), &cbi[add_cw[i]], bias[i].size()*sizeof(T), cudaMemcpyDeviceToHost);
            
        }
        cudaMemcpy(y.data(), cy, y.size()*sizeof(T), cudaMemcpyDeviceToHost);
        for(int i = 0; i < model_input.getValidation().size(); i++){
            temp_validation = model_input.getValidation()[i];
            extendMatrix(); //before predict call and for every predict in batch
            predict(temp_validation, selection);
            reduceMatrix(); //after predict call and for every predict in batch
            resetVector(z);
            index_max_element_target = 0;
            float temp_1 = model_output.getOutputValidation()[i][0];
            for(int q =1; q<model_output.getOutputValidation()[i].size(); q++){
                if(model_output.getOutputValidation()[i][q] > temp_1 ){
                    index_max_element_target = q;
                    temp_1 = model_output.getOutputValidation()[i][q];
                }
            }
            index_max_element_train = 0;
            float temp_3 = y[0];
            for(int q =1; q<y.size(); q++){
                if(y[q] > temp_3 ){
                    index_max_element_train = q;
                    temp_3 = y[q];
                }
            }
            if(index_max_element_target == index_max_element_train){
                correct_validation++;
            }
            operations_validation++;
        }
        validation_accuracy = (float)correct_validation/operations_validation;
        outputFile << "  validation Accuracy: " << std::setw(9) << validation_accuracy ;
        outputFile << "  time: " << dt_01 << " ms" << std::endl << std::flush;
        accuracyCSV << epoch << "," << train_accuracy << "," << validation_accuracy << std::endl;
        //accuracyCSV << std::endl;
    }
    outputFile << std::endl;
    outputFile << "operations: " << operations << std::endl;
    std::cout << std::endl;
    //profileFile << layers[0].getNeurons() << "," << layers[1].getNeurons() << "," << layers[2].getNeurons() << "," << cuda_block_size << ",";
    for(int i = 0; i < times.size(); i++){
        //std::cout << "time " << i << ": " << (float)times[i]/total_opp << " mics, total: " << times[i] << std::endl;
        //profileFile << (float)times[i]/total_opp << ",";
    }
    std::cout << std::endl;
    for(int i = 0; i < times2.size(); i++){
        //std::cout << "time2 " << i << ": " << (float)times2[i]/total_opp << " mics, total: " << times2[i] << std::endl;
        //profileFile << (float)times2[i]/total_opp << ",";
    }
    profileFile << "2" << std::endl;
    std::cout << "Total mul: " << total_opp << std::endl;

    //evaluating accuracy on test set
    std::vector<T> temp_test;
    int correct_test = 0;
    int operations_test = 0;
    for(int i = 0; i < model_input.getTest().size(); i++){
        temp_test = model_input.getTest()[i];
        extendMatrix(); //before predict call and for every predict in batch
        predict(temp_test, selection);
        reduceMatrix(); //after predict call and for every predict in batch
        resetVector(z);
        
        index_max_element_target = 0;
        float temp_1 = model_output.getOutputTest()[i][0];
        for(int q =1; q<model_output.getOutputTest()[i].size(); q++){
            if(model_output.getOutputTest()[i][q] > temp_1 ){
                index_max_element_target = q;
                temp_1 = model_output.getOutputTest()[i][q];
            }
        }
        index_max_element_train = 0;
        float temp_3 = y[0];
        for(int q =1; q<y.size(); q++){
            if(y[q] > temp_3 ){
                index_max_element_train = q;
                temp_3 = y[q];
            }
        }
        if(index_max_element_target == index_max_element_train){
            correct_test++;
        }
        operations_test++;
    }
    float test_accuracy = (float)correct_test/operations_test;
    outputFile << std::endl;
    outputFile << "Final Accuracy on the TestSet: " << test_accuracy << std::endl;
    outputFile << std::endl;
    std::cout << std::endl;
    std::cout << "Final Accuracy on the TestSet: " << test_accuracy << std::endl;
    outputFile << std::endl;
    const auto t1_0 = std::chrono::high_resolution_clock::now();
    int64_t dt_00 = std::chrono::duration_cast<std::chrono::milliseconds>(t1_0 - t0_0).count();
    std::cout << std::endl;
    std::cout << std::endl;
    std::cout << "Train and evaluation on Test-Set successfully completed in " << (float)dt_00/1000 << " sec !" << std::endl;
    std::cout << std::endl;
    outputFile.close(); 
    accuracyCSV.close();
    lossCSV.close();
    profileFile.close(); 

    freeCudaPointers();


}
template void Model<float>::train(int& selection, int flag);
template void Model<double>::train(int& selection, int flag);




//****************************************************************************************************************************************************
/**
 * This function defined in Model.hpp allocate the memory for the cuda pointers and copy the data from the host to the device
 **/

template<typename T>
void Model<T>::setCudaPointers(){
    int weights_size=0;
    for (int i=0; i<weights.size(); i++){
        weights_size += weights[i].size();
    }
    add_cw.push_back(0);
    add_tmp_cw.push_back(0);
    add_trsp_cw.push_back(0);
    cudaMallocManaged((void **) &cw, sizeof(T)*weights_size);
    cudaMallocManaged((void **) &tmp_cw, sizeof(T)*weights_size);
    cudaMallocManaged((void **) &trsp_cw, sizeof(T)*weights_size);
    int index_shift = 0;
    for(int i = 0; i < weights.size(); i++){
        cudaMemcpy(cw+index_shift, weights[i].data(), weights[i].size()*sizeof(T), cudaMemcpyHostToDevice);
        cudaMemcpy(tmp_cw+index_shift, weights[i].data(), weights[i].size()*sizeof(T), cudaMemcpyHostToDevice);
        cudaMemcpy(trsp_cw+index_shift, weights[i].data(), weights[i].size()*sizeof(T), cudaMemcpyHostToDevice);
        index_shift += weights[i].size();
        if (i < weights.size()-1){
            add_cw.push_back(index_shift);
            add_tmp_cw.push_back(index_shift);
            add_trsp_cw.push_back(index_shift);
        } 
    }
    for(int i=0; i<weights_size; i++){
        tmp_cw[i] = 0;
    }
    //std::cout << std::endl;
    /**std::cout << "cw: " << std::endl;
    for(int i = 0; i < weights.size(); i++){
        for(int j = 0; j < weights[i].size(); j++){
            std::cout << cw[j+add_cw[i]] << " ";   
        }
        std::cout << std::endl;
    }**/
    //cudaFree(cw);
    //cudaFree(tmp_cw);


    add_ch.push_back(0);
    int h_size=0;
    for (int i=0; i<h.size(); i++){
        h_size += h[i].size();
        if (i < h.size()-1){
            add_ch.push_back(h_size);
        }
    }
    // std::cout << "add_ch: " << std::endl;
    // for(int i = 0; i < add_ch.size(); i++){
    //     std::cout << add_ch[i] << " ";
    // }
    // std::cout << std::endl;
    cudaMallocManaged((void **) &ch, sizeof(T)*h_size);
    index_shift = 0;
    for(int i = 0; i < h.size(); i++){
        cudaMemcpy(ch+index_shift, h[i].data(), h[i].size()*sizeof(T), cudaMemcpyHostToDevice);
        index_shift += h[i].size();
    }
    //cudaFree(ch);

    add_cz.push_back(0);
    int z_size =0;
    for (int i=0; i<z.size(); i++){
        z_size += z[i].size();
        if (i < z.size()-1){
            add_cz.push_back(z_size);
        }
    }
    // std::cout << "add_cz: " << std::endl;
    // for(int i = 0; i < add_cz.size(); i++){
    //     std::cout << add_cz[i] << " ";
    // }
    // std::cout << std::endl;
    cudaMallocManaged((void **) &cz, sizeof(T)*z_size);
    index_shift = 0;
    for(int i = 0; i < z.size(); i++){
        cudaMemcpy(cz+index_shift, z[i].data(), z[i].size()*sizeof(T), cudaMemcpyHostToDevice);
        index_shift += z[i].size();
    }
    //cudaFree(cz);

    add_cdE_db.push_back(0);
    int dE_db_size=0;
    for (int i=0; i<dE_db.size(); i++){
        dE_db_size += dE_db[i].size();
        if (i < dE_db.size()-1){
            add_cdE_db.push_back(dE_db_size);
        }
    }
    // std::cout << "add_cdE_db: " << std::endl;
    // for(int i = 0; i < add_cdE_db.size(); i++){
    //     std::cout << add_cdE_db[i] << " ";
    // }
    // std::cout << std::endl;
    cudaMallocManaged((void **) &cdE_db, sizeof(T)*dE_db_size);
    index_shift = 0;
    for(int i = 0; i < dE_db.size(); i++){
        cudaMemcpy(cdE_db+index_shift, dE_db[i].data(), dE_db[i].size()*sizeof(T), cudaMemcpyHostToDevice);
        index_shift += dE_db[i].size();
    }
    //cudaFree(cdE_db);

    add_cdE_dw.push_back(0);
    int dE_dw_size=0;
    for (int i=0; i<dE_dw.size(); i++){
        dE_dw_size += dE_dw[i].size();
        if (i < dE_dw.size()-1){
            add_cdE_dw.push_back(dE_dw_size);
        }
    }
    // std::cout << "add_cdE_dw: " << std::endl;
    // for(int i = 0; i < add_cdE_dw.size(); i++){
    //     std::cout << add_cdE_dw[i] << " ";
    // }
    // std::cout << std::endl;
    cudaMallocManaged((void **) &cdE_dw, sizeof(T)*dE_dw_size);
    index_shift = 0;
    for(int i = 0; i < dE_dw.size(); i++){
        cudaMemcpy(cdE_dw+index_shift, dE_dw[i].data(), dE_dw[i].size()*sizeof(T), cudaMemcpyHostToDevice);
        index_shift += dE_dw[i].size();
    }
    //cudaFree(cdE_dw);

    add_cdE_dx.push_back(0);
    int dE_dx_size=0;
    for (int i=0; i<dE_dx.size(); i++){
        dE_dx_size += dE_dx[i].size();
        if (i < dE_dx.size()-1){
            add_cdE_dx.push_back(dE_dx_size);
        }
    }
    // std::cout << "add_cdE_dx: " << std::endl;
    // for(int i = 0; i < add_cdE_dx.size(); i++){
    //     std::cout << add_cdE_dx[i] << " ";
    // }
    // std::cout << std::endl;
    cudaMallocManaged((void **) &cdE_dx, sizeof(T)*dE_dx_size);
    index_shift = 0;
    for(int i = 0; i < dE_dx.size(); i++){
        cudaMemcpy(cdE_dx+index_shift, dE_dx[i].data(), dE_dx[i].size()*sizeof(T), cudaMemcpyHostToDevice);
        index_shift += dE_dx[i].size();
    }
    //cudaFree(cdE_dx);

    add_in.push_back(0);
    int in_size = model_input.getTrain()[0].size();
    
    // std::cout << "add_in: " << std::endl;
    // std::cout << add_in[0] << " " << std::endl;
    
    cudaMallocManaged((void **) &in, sizeof(T)*in_size);
    //cudaFree(in);

    add_cbi.push_back(0);
    add_tmp_cbi.push_back(0);
    int bi_size=0;
    for (int i=0; i<bias.size(); i++){
        bi_size += bias[i].size();
        if (i < bias.size()-1){
            add_cbi.push_back(bi_size);
            add_tmp_cbi.push_back(bi_size);
        }
    }
    // std::cout << "add_cbi: " << std::endl;
    // for(int i = 0; i < add_cbi.size(); i++){
    //     std::cout << add_cbi[i] << " ";
    // }
    // std::cout << std::endl;
    cudaMallocManaged((void **) &cbi, sizeof(T)*bi_size);
    cudaMallocManaged((void **) &tmp_cbi, sizeof(T)*bi_size);
    index_shift = 0;
    for(int i = 0; i < bias.size(); i++){
        cudaMemcpy(cbi+index_shift, bias[i].data(), bias[i].size()*sizeof(T), cudaMemcpyHostToDevice);
        cudaMemcpy(tmp_cbi+index_shift, bias[i].data(), bias[i].size()*sizeof(T), cudaMemcpyHostToDevice);
        index_shift += bias[i].size();
    }
    for(int i=0; i<=bi_size;i++){
        tmp_cbi[i] = 0;
    }
    //cudaFree(cbi);
    //cudaFree(tmp_cbi);

    add_cy.push_back(0);
    int y_size = y.size();
    // std::cout << "add_cy: " << std::endl;
    // std::cout << add_cy[0] << " " << std::endl;
    cudaMallocManaged((void **) &cy, sizeof(T)*y_size);
    cudaMemcpy(cy+index_shift, y.data(), y.size()*sizeof(T), cudaMemcpyHostToDevice);

    //cudaFree(cy);

    add_cdAct_z.push_back(0);
    int dAct_z_size=0;
    for (int i=0; i<dAct_z.size(); i++){
        dAct_z_size += dAct_z[i].size();
        if (i < dAct_z.size()-1){
            add_cdAct_z.push_back(dAct_z_size);
        }
    }
    // std::cout << "add_cdAct_z: " << std::endl;
    // for(int i = 0; i < add_cdAct_z.size(); i++){
    //     std::cout << add_cdAct_z[i] << " ";
    // }
    // std::cout << std::endl;
    cudaMallocManaged((void **) &cdAct_z, sizeof(T)*dAct_z_size);
    index_shift = 0;
    for(int i = 0; i < dAct_z.size(); i++){
        cudaMemcpy(cdAct_z+index_shift, dAct_z[i].data(), dAct_z[i].size()*sizeof(T), cudaMemcpyHostToDevice);
        index_shift += dAct_z[i].size();
    }
    //cudaFree(cdAct_z);

    add_cdE_dy.push_back(0);
    int dE_dy_size = model_output.getShapeOutputData();
    // std::cout << "add_cdE_dy: " << std::endl;
    // std::cout << add_cdE_dy[0] << " " << std::endl;
    cudaMallocManaged((void **) &cdE_dy, sizeof(T)*dE_dy_size);
    cudaMemcpy(cdE_dy, dE_dy.data(), dE_dy.size()*sizeof(T), cudaMemcpyHostToDevice);
    //cudaFree(cdE_dy);




}
template void Model<float>::setCudaPointers();
template void Model<double>::setCudaPointers();

//****************************************************************************************************************************************************
/**
 * This function defined in Model.hpp free the memory of the cuda pointers
 **/

template<typename T>
void Model<T>::freeCudaPointers(){
    cudaFree(cw);
    cudaFree(ch);
    cudaFree(cz);
    cudaFree(cdE_db);
    cudaFree(cdE_dw);
    cudaFree(cdE_dx);
    cudaFree(in);
    cudaFree(cbi);
    cudaFree(cy);
    cudaFree(cdAct_z);
    cudaFree(cdE_dy);
    cudaFree(tmp_cw);
    cudaFree(tmp_cbi);
    cudaFree(trsp_cw);
}
template void Model<float>::freeCudaPointers();
template void Model<double>::freeCudaPointers();

