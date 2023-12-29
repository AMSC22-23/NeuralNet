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

    auto result = readIrisData<float>("Iris.csv");
    auto split_result = getIrisSets<float>(result, 0.6, 0.2, 0.2);
    //auto split_result = getIrisSets<float>(result, 0.5, 0.2, 0.3);

    //RETRIVING .CSV DATA
    trainSet = std::get<0>(split_result);
    trainOut = std::get<1>(split_result);
    validationSet = std::get<2>(split_result);
    validationOut = std::get<3>(split_result);
    testSet = std::get<4>(split_result);
    testOut = std::get<5>(split_result);

    /**std::cout << trainSet.size() << std::endl;
    std::cout << trainSet[0].size() << std::endl;
    for(int i = 0; i<trainSet.size(); i++){
        for(int j = 0; j<trainSet[0].size(); j++){
            std::cout << trainSet[i][j] << " ";
        }
        std::cout << std::endl;
    }**/

   

    //CREATING MODEL
    shuffleData(trainSet, trainOut);
    Input input(trainSet, validationSet, testSet);
    Output output(trainOut, validationOut, testOut, "sigmoid");
    Layer layer1("prova", 128, "ReLu"), layer2("prova2", 70, "ReLu"), layer3("prova3", 10, "ReLu");  //best in train set at the moment
    Model model("Modello",100, 16, 0.05, "MSE", input, output, "early_stop"); //batch around 8-16 learning rate 0.05 works well
    model.setWeightdInitialization("He");  //He best in train set at the moment, Xavier works well too, Normal is fine, Uniform do not work

    /**Output output(trainOut, validationOut, testOut, "ReLu");
    Layer layer1("prova", 5, "ReLu"), layer2("prova2", 70, "ReLu"), layer3("prova3", 10, "ReLu");
    Model model("Modello",100, 16, 0.05, "MSE", input, output, "early_stop");
    model.setWeightdInitialization("debug");**/
    
    

    //BUILDING THE MODEL
    
    model.addLayer(layer1);
    //model.addLayer(layer2);
    //model.addLayer(layer3);

    

    model.buildModel();

    //model.printAllWeightsToFile(); //DEBUG

    //TRAINING THE MODEL
    const auto t0 = std::chrono::high_resolution_clock::now();
    model.train( a);
    const auto t1 = std::chrono::high_resolution_clock::now();
    int64_t dt_01 = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();

    std::cout << "Duration of the entire training: " << dt_01 << " ms" << std::endl;


    //Debug test

    //std::vector<float> test_input = {1,1,1,1};
    

    //model.predict(test_input, a, 1);
    //model.printAllWeightsToFile();

    return 0;
}
