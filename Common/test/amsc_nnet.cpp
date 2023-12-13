//#include "../include/network.hpp"
#include "../include/model.hpp"


void genFakeData(std::vector<std::vector<float>>& a, int rows, int cols){
    for (int i = 0; i < rows; ++i) {
        std::vector<float> row;
        for (int j = 0; j < cols; ++j) {
            row.push_back(1.0f * (j + 1));  // Aggiungi [1, 2, 3]
        }
        a.push_back(row);
    }

}

int main(){

    std::vector<std::vector<float>> trainSet, validationSet, testSet, trainOut, validationOut, testOut;

    genFakeData(trainSet, 100, 5);
    genFakeData(validationSet, 50, 5);
    genFakeData(testSet, 20, 5);
    genFakeData(trainOut, 100, 3);
    genFakeData(validationOut, 50, 3);
    genFakeData(testOut, 20, 3);
    
    int a = 0;
    Input input(trainSet, validationSet, testSet);
    Output output(trainOut, validationOut, testOut, "sigmoid");
    Layer layer1("prova", 3, "sigmoid"), layer2("prova2", 7, "sigmoid"), layer3("prova3", 10, "sigmoid");
    Model model("Modello",100, 10, 0.01, "MSE", input, output, "early_stop");
    model.addLayer(layer1);
    model.addLayer(layer2);
    model.addLayer(layer3);


    Input<float> input2(model.getInput());
    Output<float> output2(model.getOutput());

    /**input.printShape();
    std::cout << std::endl;
    output.printShape();
    std::cout << std::endl;
    input2.printShape();
    std::cout << std::endl;
    layer1.printLayer();
    std::cout << std::endl;
    model.printLayers();**/

    model.buildModel();

    model.printWeigts();

    model.predict(trainSet[0], a);
  



    return 0;
}