#include "../include/model.hpp"
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
    mul_funct(input, weights[0], z[0], 1, weights_shape[0][0]+1, weights_shape[0][1], matrix_mul_optimisation, cuda_block_size);   //weights[0], input, z, weights_shape[0][0], weights_shape[0][1], 1, matrix_mul_optimisation);
    activationFun(z[0], h[0], layers[0].getActFun());
    
    for(int loop = 0; loop < layers.size(); loop++){
        mul_funct(h[loop], weights[loop+1], z[loop+1], 1, weights_shape[loop+1][0]+1, weights_shape[loop+1][1], matrix_mul_optimisation, cuda_block_size);
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
    mul_funct(input, weights[0], z[0], 1, weights_shape[0][0]+1, weights_shape[0][1], matrix_mul_optimisation, cuda_block_size);   //weights[0], input, z, weights_shape[0][0], weights_shape[0][1], 1, matrix_mul_optimisation);
    activationFun(z[0], h[0], layers[0].getActFun());
    
    for(int loop = 0; loop < layers.size(); loop++){
        mul_funct(h[loop], weights[loop+1], z[loop+1], 1, weights_shape[loop+1][0]+1, weights_shape[loop+1][1], matrix_mul_optimisation, cuda_block_size);
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


//****************************************************************************************************************************************************
//This function defined in Model.hpp compute the backpropagation of the model using the chain rule and Gradient Descent

template<typename T>
void Model<T>::backPropagation(std::vector<T>& input, std::vector<T>& dE_dy, int& selection){
    int one=1;
    std::vector<T> temp;
    activationFunDerivative(z[layers.size()], dAct_z[layers.size()], model_output.getOutputAct_fun());
    dE_db[layers.size()] = mul(dE_dy, dAct_z[layers.size()]);
    temp = transposeMatrix(h[layers.size()-1], one, h[layers.size()-1].size());
    mul_funct(temp , dE_db[layers.size()],dE_dw[layers.size()], h[layers.size()-1].size(), one, dE_db[layers.size()].size(), matrix_mul_optimisation, cuda_block_size);
    transposeMatrix2(weights[layers.size()], temp, weights_shape[layers.size()][0], weights_shape[layers.size()][1]);
    mul_funct(dE_db[layers.size()], temp, dE_dx[layers.size()-1], one,  dE_db[layers.size()].size(), weights_shape[layers.size()][0], matrix_mul_optimisation, cuda_block_size);
    for (int i=layers.size()-1; i > 0; i--){
        activationFunDerivative(z[i], dAct_z[i], layers[i].getActFun());
        dE_db[i] = mul(dE_dx[i], dAct_z[i]);
        temp = transposeMatrix(h[i-1], one, h[i-1].size());
        mul_funct(temp, dE_db[i], dE_dw[i], h[i-1].size(), one, dE_db[i].size(), matrix_mul_optimisation, cuda_block_size);
        transposeMatrix2(weights[i], temp, weights_shape[i][0], weights_shape[i][1]);
        mul_funct(dE_db[i], temp, dE_dx[i-1], one,  dE_db[i].size(), weights_shape[i][0], matrix_mul_optimisation, cuda_block_size);
    }
    activationFunDerivative(z[0], dAct_z[0], layers[0].getActFun());
    dE_db[0] = mul(dE_dx[0], dAct_z[0]);
    temp = transposeMatrix(input, one, input.size());
    mul_funct(temp, dE_db[0], dE_dw[0], input.size(), one, dE_db[0].size(), matrix_mul_optimisation, cuda_block_size);
    
}
template void Model<float>::backPropagation(std::vector<float>& input, std::vector<float>& dE_dy, int& selection);
template void Model<double>::backPropagation(std::vector<double>& input, std::vector<double>& dE_dy, int& selection);


//****************************************************************************************************************************************************
/**
 * This function defined in Model.hpp take as input the chosen matrix multiplication algorithm chosen with "selection"
 * and train the parameters of the model using predict and backpropagation function
 **/

template<typename T>
void Model<T>::train(int& selection){
    std::ofstream outputFile("Train_Output.txt");
    std::ofstream accuracyCSV("Accuracy.csv");
    std::ofstream lossCSV("Loss.csv");
    accuracyCSV << "epoch, train_accuracy, validation_accuracy" << std::endl;
    lossCSV << "epoch, batch, loss" << std::endl;
    std::vector<std::vector<T>> tempWeights = createTempWeightMAtrix(weights);
    std::vector<std::vector<T>> tempBias = createTempBiasMAtrix(bias);
    int batch = model_input.getTrain().size() / model_batch_size;
    int count=0, operations = 0, correct = 0;
    float maxElement_train, max_element_target;
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
                    temp = model_input.getTrain()[batch_loop*model_batch_size+i];
                    extendMatrix();         //before predict call and for every predict in batch
                    predict(temp, selection);
                    reduceMatrix();          //after predict call and for every predict in batch
                    applyLossFunction(y, model_output.getOutputTrain()[batch_loop*model_batch_size+i], dE_dy, model_loss_fun);
                    backPropagation(temp, dE_dy, selection);
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
    }
    outputFile << std::endl;
    outputFile << "operations: " << operations << std::endl;

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
}

template void Model<float>::train(int& selection);
template void Model<double>::train(int& selection);