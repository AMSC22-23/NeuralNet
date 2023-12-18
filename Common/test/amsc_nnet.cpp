//#include "../include/network.hpp"
#include "../include/model.hpp"
#include <fstream>
#include <sstream>
#include <tuple>

void genFakeData(std::vector<std::vector<float>>& a, int rows, int cols){
    for (int i = 0; i < rows; ++i) {
        std::vector<float> row;
        for (int j = 0; j < cols; ++j) {
            row.push_back(1.0f * (j + 1));  // Aggiungi [1, 2, 3]
        }
        a.push_back(row);
    }
}

// Tuple to represent a row of Iris data
// Each tuple contains 4 doubles (SepalLengthCm, SepalWidthCm, PetalLengthCm, PetalWidthCm) 
// and a tuple representing the species in one-hot encoding format
using IrisTuple = std::tuple<float, float, float, float, std::tuple<int, int, int>>;



// Function to normalize a float value between min and max
float normalize(float value, float min, float max) {
    return (value - min) / (max - min);
}


// Function to read data from Iris.CSV file and split it into vectors based on species
std::tuple<std::vector<IrisTuple>, std::vector<IrisTuple>, std::vector<IrisTuple>> readIrisData(const std::string& file_name) {
    std::ifstream file(file_name);

    // Check if the file is opened successfully
    if (!file.is_open()) {
        std::cerr << "Error opening the CSV file." << std::endl;
        exit(1);
    }

    // Vectors to store data for different species
    std::vector<IrisTuple> setosa_data;
    std::vector<IrisTuple> versicolor_data;
    std::vector<IrisTuple> virginica_data;


    //Matrix to store all numerical input side data
    std::vector<std::vector<float>> AllData;

    //floats to store maximum and minimum values
    float min = 1000000.0f;
    float max = 0.0f;

    // Read the file line by line
    std::string line;
    bool firstLine = true;  // Added to skip the first line
    while (getline(file, line)) {
        if (firstLine) {
            // Skip the first line (header)
            firstLine = false;
            continue;
        }

        std::stringstream ss(line);
        std::string field;

        // Ignore the ID
        getline(ss, field, ',');

        // Save in 'field' the next characters until the comma
        getline(ss, field, ',');
        // Convert field to float and save the value in sepal_length
        float sepal_length = stof(field);
        //update min and max
        min = std::min(sepal_length, min);
        max = std::max(sepal_length, max);

        getline(ss, field, ',');
        float sepal_width = stof(field);
        min = std::min(sepal_width, min);
        max = std::max(sepal_width, max);

        getline(ss, field, ',');
        float petal_length = stof(field);
        min = std::min(petal_length, min);
        max = std::max(petal_length, max);

        getline(ss, field, ',');
        float petal_width = stof(field);
        min = std::min(petal_width, min);
        max = std::max(petal_width, max);

        getline(ss, field, ',');
        float speciesFloat;

        if (field == "Iris-setosa") {
            speciesFloat = 1.0f;

        } else if (field == "Iris-versicolor") {
            speciesFloat = 2.0f;

        } else if (field == "Iris-virginica") {
            speciesFloat = 3.0f;
        }

        std::vector<float> lineData = {sepal_length, sepal_width, petal_length, petal_width, speciesFloat};
        AllData.emplace_back(lineData);

    }

    // Close the file
    file.close();

    for (const auto& vect : AllData) {

        IrisTuple data;
        std::tuple<int, int, int> species_tuple;

        auto normalizedSL = normalize(vect[0], min, max);
        auto normalizedSW = normalize(vect[1], min, max);
        auto normalizedPL = normalize(vect[2], min, max);
        auto normalizedPW = normalize(vect[3], min, max);
        auto code = vect[4];

        // Represent the species in one-hot encoding format
        if (code == 1.0f) {
            species_tuple = std::make_tuple(1, 0, 0);
            data = std::make_tuple(normalizedSL, normalizedSW, normalizedPL, normalizedPW, species_tuple);
            setosa_data.emplace_back(data);
        } else if (code == 2.0f) {
            species_tuple = std::make_tuple(0, 1, 0);
            data = std::make_tuple(normalizedSL, normalizedSW, normalizedPL, normalizedPW, species_tuple);
            versicolor_data.emplace_back(data);
        } else if (code == 3.0f) {
            species_tuple = std::make_tuple(0, 0, 1);
            data = std::make_tuple(normalizedSL, normalizedSW, normalizedPL, normalizedPW, species_tuple);
            virginica_data.emplace_back(data);
        }
    }

    // Return the tuple containing the three vectors of tuples
    return std::make_tuple(setosa_data, versicolor_data, virginica_data);
}

// Function to split the data into a training set, validation set, and test set
// Returns a tuple containing six vectors of floats
std::tuple<std::vector<std::vector<float>>, std::vector<std::vector<float>>,
           std::vector<std::vector<float>>, std::vector<std::vector<float>>,
           std::vector<std::vector<float>>, std::vector<std::vector<float>>>
getIrisSets(const std::tuple<std::vector<IrisTuple>, std::vector<IrisTuple>, std::vector<IrisTuple>>& iris_data,
          double train_ratio, double val_ratio, double test_ratio) {

    // Check that the ratios are correct
    double total_ratio = train_ratio + val_ratio + test_ratio;

    if (total_ratio != 1.0) {
        std::cerr << "Error: total ratio must be 1.0." << std::endl;
        exit(1);
    }

    // Create training, validation, and test sets
    std::vector<std::vector<float>> train_set, train_out, val_set, val_out, test_set, test_out;

    auto iris_array = {std::get<0>(iris_data), std::get<1>(iris_data), std::get<2>(iris_data)};

    // Loop over each vector in the iris_array
    for (const auto& species_data : iris_array) {

        // Calculate the number of elements for each set
        std::size_t total_size = species_data.size();
        std::size_t train_size = static_cast<std::size_t>(train_ratio * total_size);
        std::size_t val_size = static_cast<std::size_t>(val_ratio * total_size);
        std::size_t test_size = total_size - train_size - val_size;

        for (size_t i = 0; i < total_size; ++i) {
            std::vector<float> inputSide = {std::get<0>(species_data[i]), std::get<1>(species_data[i]), std::get<2>(species_data[i]), std::get<3>(species_data[i])};
            std::vector<float> outputSide = {static_cast<float>(std::get<0>(std::get<4>(species_data[i]))),
                                             static_cast<float>(std::get<1>(std::get<4>(species_data[i]))),
                                             static_cast<float>(std::get<2>(std::get<4>(species_data[i])))};

            if (i < train_size) {
                train_set.push_back(inputSide);
                train_out.push_back(outputSide);
            } else if (i < train_size + val_size) {
                val_set.push_back(inputSide);
                val_out.push_back(outputSide);
            } else {
                test_set.push_back(inputSide);
                test_out.push_back(outputSide);
            }
        }
    }

    // Return the tuple containing the three sets
    return std::make_tuple(train_set, train_out, val_set, val_out, test_set, test_out);
}


int main(){

    std::vector<std::vector<IrisTuple>> iris_se_data, iris_vi_data, iris_ve_data;
    std::vector<std::vector<float>> trainSet, validationSet, testSet, trainOut, validationOut, testOut;
    int a=0;

    auto result = readIrisData("Iris.csv");
    auto split_result = getIrisSets(result, 0.6, 0.2, 0.2);

    trainSet = std::get<0>(split_result);
    trainOut = std::get<1>(split_result);
    validationSet = std::get<2>(split_result);
    validationOut = std::get<3>(split_result);
    testSet = std::get<4>(split_result);
    testOut = std::get<5>(split_result);

    

    // Ottenere i vettori dalla tupla risultante
    /**std::vector<IrisTuple> vector1 = std::get<0>(result);
    std::vector<IrisTuple> vector2 = std::get<1>(result);
    std::vector<IrisTuple> vector3 = std::get<2>(result);

    std::cout << "Setosa: " << vector1.size() << std::endl;
    std::cout << "Versicolor: " << vector2.size() << std::endl;
    std::cout << "Virginica: " << vector3.size() << std::endl;

    // Ottenere i vettori di tuple dalla tupla risultante
    std::tuple<int, int, int> target1 = std::get<4>(vector1[0]);
    std::cout << "Target: " << std::get<0>(target1) << " " << std::get<1>(target1) << " " << std::get<2>(target1) << std::endl;


    std::vector<std::vector<float>> firstFourElements;
    std::vector<std::vector<float>> tupleElements;

    for (const auto& iris : vector1) {
        std::vector<float> firstFour = {std::get<0>(iris), std::get<1>(iris), std::get<2>(iris), std::get<3>(iris)};
        firstFourElements.push_back(firstFour);

        const auto& innerTuple = std::get<4>(iris);
        std::vector<float> tupleValues = {static_cast<float>(std::get<0>(innerTuple)),
                                          static_cast<float>(std::get<1>(innerTuple)),
                                          static_cast<float>(std::get<2>(innerTuple))};
        tupleElements.push_back(tupleValues);
    }

    for (const auto& iris : vector2) {
        std::vector<float> firstFour = {std::get<0>(iris), std::get<1>(iris), std::get<2>(iris), std::get<3>(iris)};
        firstFourElements.push_back(firstFour);

        const auto& innerTuple = std::get<4>(iris);
        std::vector<float> tupleValues = {static_cast<float>(std::get<0>(innerTuple)),
                                          static_cast<float>(std::get<1>(innerTuple)),
                                          static_cast<float>(std::get<2>(innerTuple))};
        tupleElements.push_back(tupleValues);
    }

    for (const auto& iris : vector3) {
        std::vector<float> firstFour = {std::get<0>(iris), std::get<1>(iris), std::get<2>(iris), std::get<3>(iris)};
        firstFourElements.push_back(firstFour);

        const auto& innerTuple = std::get<4>(iris);
        std::vector<float> tupleValues = {static_cast<float>(std::get<0>(innerTuple)),
                                          static_cast<float>(std::get<1>(innerTuple)),
                                          static_cast<float>(std::get<2>(innerTuple))};
        tupleElements.push_back(tupleValues);
    }**/

    /**std::cout << "First four elements: " << firstFourElements.size() << std::endl;
    std::cout << "Tuple elements: " << tupleElements.size() << std::endl;

    std::cout << "First four elements: " << std::endl;
    for (const auto& element : firstFourElements) {
        std::cout << element[0] << " " << element[1] << " " << element[2] << " " << element[3] << std::endl;
    }
    std::cout << std::endl;
    std::cout << "Tuple elements: " << std::endl;
    for (const auto& element : tupleElements) {
        std::cout << element[0] << " " << element[1] << " " << element[2] << std::endl;
    }**/  


    // Creare i set di addestramento, validazione e test
    //auto iris_sets = getIrisSets(result, 0.6, 0.2, 0.2);  



    /**genFakeData(trainSet, 101, 5);
    genFakeData(validationSet, 50, 5);
    genFakeData(testSet, 20, 5);
    genFakeData(trainOut, 101, 3);
    genFakeData(validationOut, 50, 3);
    genFakeData(testOut, 20, 3);**/

    //Input input(trainSet, validationSet, testSet);
    Input input(trainSet, validationSet, testSet);
    //Output output(trainOut, validationOut, testOut, "sigmoid");
    Output output(trainOut, validationOut, testOut, "sigmoid");
    Layer layer1("prova", 3, "ReLu"), layer2("prova2", 7, "ReLu"), layer3("prova3", 10, "ReLu");
    Model model("Modello",100, 10, 0.1, "MSE", input, output, "early_stop");
    std::vector<float> faketest = {0.5,0.6,0.8};
    model.addLayer(layer1);
    model.addLayer(layer2);
    model.addLayer(layer3);


    //Input<float> input2(model.getInput());
    //Output<float> output2(model.getOutput());

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

    //model.predict(trainSet[0], a, 1);

    //model.extendMatrix();
    //model.predict(trainSet[0], a);
    //model.reduceMatrix();

    //model.backPropagation(trainSet[0], faketest, a);

    model.train( a);
  



    return 0;
}
