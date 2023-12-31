//#include "../include/network.hpp"
#include "../include/irisLoader.hpp"
#include "../include/model.hpp"
#include <fstream>
#include <sstream>
#include <tuple>

#include "../include/functions_utilities.hpp"
#include <algorithm>
#include <random>
#include <chrono>



int main(){

    using IrisTuple = std::tuple<float, float, float, float, std::tuple<int, int, int>>;

    std::vector<std::vector<IrisTuple>> iris_se_data, iris_vi_data, iris_ve_data;
    std::vector<std::vector<float>> trainSet, validationSet, testSet, trainOut, validationOut, testOut;
    int a=0;

    auto result = readIrisData<float>("./DataSet/Iris.csv");
    auto split_result = getIrisSets<float>(result, 0.6, 0.2, 0.2);

    //RETRIVING .CSV DATA
    trainSet = std::get<0>(split_result);
    trainOut = std::get<1>(split_result);
    validationSet = std::get<2>(split_result);
    validationOut = std::get<3>(split_result);
    testSet = std::get<4>(split_result);
    testOut = std::get<5>(split_result);

    

    //CREATING MODEL
    shuffleData(trainSet, trainOut);
    Input input(trainSet, validationSet, testSet);
    Output output(trainOut, validationOut, testOut, "sigmoid");
    Layer layer1("layer1", 5, "ReLu"), layer2("layer2", 200, "ReLu"), layer3("layer33", 300, "ReLu");  //128 neurons and one layer best in train set at the moment
    Model model("myModel",100, 16, 0.05, "MSE", input, output, "early_stop"); //batch around 8-16 learning rate 0.05 works well
    model.setWeightdInitialization("He");  //He best in train set at the moment, Xavier works well too, Normal is fine, Uniform do not work

    
    //BUILDING THE MODEL
    
    model.addLayer(layer1);
    //model.addLayer(layer2);
    //model.addLayer(layer3);

    model.buildModel();
    model.printAllWeightsToFile();
    


    //TRAINING THE MODEL
    
    //model.train( a);
    
    //Debug test

    model.predict(trainSet[0], a, 1);
    model.backPropagation(trainSet[0], trainOut[0], a);
    model.printAllWeightsToFile();

    //model.printAllWeightsToFile();

    return 0;
}
