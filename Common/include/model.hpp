#include "network.hpp"
#include <fstream>

template<typename T>
class Model{
    public:
    Model(std::string name, int epochs, int batch_size, float learning_rate, std::string loss_fun, Input<T> input, Output<T> output, std::string stop_cryteria):
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

    //@note: method should be const
    void printWeigts() const;

    void printModel();

    void setWeightdInitialization(const std::string weights_model){
        weights_initialisation = weights_model;
    }
    void printAllWeightsToFile();
    //@note: making a prediction should not change the input
    void predict(std::vector<T>& input, int& selection); //this version need to be called only after the resizing of the weights
    void predict(std::vector<T>& input, int& selection, int flag);
    void backPropagation(std::vector<T>& input, std::vector<T>& dE_dy, int& selection);
    void train(int& selection);
    void extendMatrix();
    void reduceMatrix();
    void initialiseVector(std::vector<std::vector<T>>& default_weights, const std::string& weights_model);
    
    //@note: method should be const
    Input<T> getInput(){return model_input;}
    Output<T> getOutput(){return model_output;}

    protected:
    std::vector<std::vector<T>> dE_dw, z, h, dAct_z, dE_dx, dE_db;
    std::vector<T> y, dE_dy;
    
    private:
    std::vector<Layer> layers;
    Input<T> model_input;
    Output<T> model_output;
    int model_epochs, model_batch_size, matrix_mul_optimisation = 0;
    float model_learning_rate;
    T default_weight = 0.3;
    std::string model_name, model_loss_fun, model_stop_cryteria, weights_initialisation = "Normal_Distribution";
    std::vector<std::vector<T>> weights, bias;
    std::vector<std::vector<int>> weights_shape;
    std::vector<T> input_layer, output_layer;
    //std::ofstream outputFile("weights.txt");
};

