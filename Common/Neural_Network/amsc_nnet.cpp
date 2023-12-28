//#include "../include/network.hpp"
#include "../include/irisLoader.hpp"
#include "../include/model.hpp"
#include <fstream>
#include <sstream>
#include <tuple>

#include "../include/functions_utilities.hpp"
#include <algorithm>
#include <random>



int main(){

    using IrisTuple = std::tuple<float, float, float, float, std::tuple<int, int, int>>;

    std::vector<std::vector<IrisTuple>> iris_se_data, iris_vi_data, iris_ve_data;
    std::vector<std::vector<float>> trainSet, validationSet, testSet, trainOut, validationOut, testOut;
    int a=0;

    auto result = readIrisData<float>("Iris.csv");
    auto split_result = getIrisSets<float>(result, 0.6, 0.2, 0.2);

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

   
    /**genFakeData(trainSet, 101, 5);   //DEBUG DATA
    genFakeData(validationSet, 50, 5);
    genFakeData(testSet, 20, 5);
    genFakeData(trainOut, 101, 3);
    genFakeData(validationOut, 50, 3);
    genFakeData(testOut, 20, 3);**/

    //CREATING MODEL
    shuffleData(trainSet, trainOut);
    Input input(trainSet, validationSet, testSet);
    Output output(trainOut, validationOut, testOut, "sigmoid");
    Layer layer1("prova", 128, "ReLu"), layer2("prova2", 70, "ReLu"), layer3("prova3", 10, "ReLu");
    Model model("Modello",100, 16, 0.05, "MSE", input, output, "early_stop");
    //std::vector<float> faketest = {0.5,0.6,0.8};
    

    //BUILDING THE MODEL
    
    model.addLayer(layer1);
    //model.addLayer(layer2);
    //model.addLayer(layer3);

    model.setWeightdInitialization("He");

    model.buildModel();

    //model.printWeigts(); //DEBUG

    //TRAINING THE MODEL
    model.train( a);

    //model.printWeigts();  //DEBUG

    

    



    return 0;
}
