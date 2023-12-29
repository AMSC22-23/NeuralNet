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

    void buildModel();/**{
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
        printLayers();
        std::cout << "Output layer: " << std::endl;
        std::cout << "Number of output introduced in the network: " << model_output.getShapeOutputData() << " activation function: " << model_output.getAct_Fun() << std::endl;
        std::cout << std::endl;

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
    }**/

    //@note: method should be const
    void printWeigts(){
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

