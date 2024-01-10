#ifndef ACTIVATION_MODEL_HPP
#define ACTIVATION_MODEL_HPP

#include "network.hpp"
#include <fstream>

//*********************************************************************************************************************

//Here the definition of the methods of the class Model, the body is defined in /src/model.cpp

//*********************************************************************************************************************

template<typename T>
class Model{
    public:
    Model(const std::string name, const int epochs, const int batch_size, const float learning_rate, const std::string loss_fun, Input<T> input, Output<T> output, const std::string stop_cryteria):
        model_name(name), model_epochs(epochs), model_batch_size(batch_size), model_learning_rate(learning_rate), model_loss_fun(loss_fun), model_stop_cryteria(stop_cryteria), model_input(input), model_output(output)
        
    {};

    void setFirstWeigts(T value){
        this->default_weight = value;
    }

    void addLayer(Layer layer){
        layers.push_back(layer);
    }

    void printLayers() const {
        std::cout << "Layers of the model: " << std::endl;
        for (int i = 0; i < layers.size(); ++i) {
            std::cout << "Layer " << i+1 << ": " << std::endl;
            layers[i].printLayer();
        }
        std::cout << std::endl;
    }

    void buildModel();

    
    void printWeigts() const;

    void printModel() const ;

    void setWeightdInitialization(const std::string weights_model){
        weights_initialisation = weights_model;
    }
    void printAllWeightsToFile() const ;
    //@note: making a prediction should not change the input***********
        //re-@note: in the way we build the function the input is resized to add the bias constant, so is not possible to declare it const
    void predict(std::vector<T>& input, const int& selection); //this version need to be called only after the resizing of the weights
    void predict(std::vector<T>& input, const int& selection, const int flag);
    void backPropagation(const std::vector<T>& input, std::vector<T>& dE_dy, const int& selection);
    void train(int& selection);
    void extendMatrix();
    void reduceMatrix();
    void initialiseVector(std::vector<std::vector<T>>& default_weights, const std::string& weights_model);
    
    Input<T> getInput() const {return model_input;}
    Output<T> getOutput() const {return model_output;}

    protected:
    std::vector<std::vector<T>> dE_dw, z, h, dAct_z, dE_dx, dE_db;
    std::vector<T> y, dE_dy;
    
    private:
    std::vector<int64_t> times;
    std::vector<Layer> layers;
    Input<T> model_input;
    Output<T> model_output;
    int model_epochs, model_batch_size, matrix_mul_optimisation = 0, cuda_block_size = 32;
    float model_learning_rate;
    T default_weight = 0.3;
    std::string model_name, model_loss_fun, model_stop_cryteria, weights_initialisation = "Normal_Distribution";
    std::vector<std::vector<T>> weights, bias;
    std::vector<std::vector<int>> weights_shape;
    std::vector<T> input_layer, output_layer;
};


#endif
