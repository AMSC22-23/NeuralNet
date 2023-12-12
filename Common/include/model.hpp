#include "network.hpp"

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

    void buildModel(){
        std::cout << "Building model..." << std::endl;
        std::cout << "Model name: " << model_name << std::endl;
        std::cout << "Number of epochs: " << model_epochs << std::endl;
        std::cout << "Batch size: " << model_batch_size << std::endl;
        std::cout << "Learning rate: " << model_learning_rate << std::endl;
        std::cout << "Loss function: " << model_loss_fun << std::endl;
        std::cout << "Stop cryteria: " << model_stop_cryteria << std::endl;
        std::cout << std::endl;

        weights.resize(layers.size()+1);
        bias.resize(layers.size()+1);
        weights_shape.resize(layers.size()+1);
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
            fillerdim = layers[block].getNeurons();
            check += 1;
        }
        std::vector<T> filler(layers[check-1].getNeurons() * model_output.getShapeOutputData(), default_weight);
        std::vector<T> fillerBias(model_output.getShapeOutputData(), default_weight);
        weights[check].resize(fillerdim * layers[check-1].getNeurons());
        weights[check] = filler;
        weights_shape[check].resize(2);
        weights_shape[check][1] = model_output.getShapeOutputData();
        weights_shape[check][0] = layers[check-1].getNeurons();
        bias[check].resize(model_output.getShapeOutputData());
        bias[check] = fillerBias;


        std::cout << "Model built!" << std::endl;
    }

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

    Input<T> getInput(){return model_input;}
    Output<T> getOutput(){return model_output;}

    private:
    std::vector<Layer> layers;
    Input<T> model_input;
    Output<T> model_output;
    int model_epochs, model_batch_size;
    float model_learning_rate;
    T default_weight = 0.3;
    std::string model_name, model_loss_fun, model_stop_cryteria;
    std::vector<std::vector<T>> weights, bias;
    std::vector<std::vector<int>> weights_shape;
    std::vector<T> input_layer, output_layer;
};