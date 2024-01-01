#include "irisLoader.hpp"
#include <fstream>
#include <sstream>
#include <tuple>
#include <iostream>

// Function to generate fake data for testing
template<typename T>
void genFakeData(std::vector<std::vector<T>>& a, int rows, int cols){
    for (int i = 0; i < rows; ++i) {
        std::vector<T> row;
        for (int j = 0; j < cols; ++j) {
            row.push_back(1.0f * (j + 1));  // Aggiungi [1, 2, 3]
        }
        a.push_back(row);
    }
}
template void genFakeData<float>(std::vector<std::vector<float>>& a, int rows, int cols);
template void genFakeData<double>(std::vector<std::vector<double>>& a, int rows, int cols);

template<typename T>
T normalize(T value, T min, T max) {
    return (value - min) / (max - min);
}
template float normalize<float>(float value, float min, float max);
template double normalize<double>(double value, double min, double max);



template<typename T>
std::tuple<std::vector<std::tuple<T, T, T, T, std::tuple<int, int, int>>>, std::vector<std::tuple<T, T, T, T, std::tuple<int, int, int>>>, std::vector<std::tuple<T, T, T, T, std::tuple<int, int, int>>>> readIrisData(const std::string& file_name) {
    using IrisTuple = std::tuple<T, T, T, T, std::tuple<int, int, int>>;
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
    std::vector<std::vector<T>> AllData;

    //floats to store maximum and minimum values
    T min = 1000000.0f;
    T max = 0.0f;

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
        // Convert field to T and save the value in sepal_length
        T sepal_length = stof(field);
        //update min and max
        min = std::min(sepal_length, min);
        max = std::max(sepal_length, max);

        getline(ss, field, ',');
        T sepal_width = stof(field);
        min = std::min(sepal_width, min);
        max = std::max(sepal_width, max);

        getline(ss, field, ',');
        T petal_length = stof(field);
        min = std::min(petal_length, min);
        max = std::max(petal_length, max);

        getline(ss, field, ',');
        T petal_width = stof(field);
        min = std::min(petal_width, min);
        max = std::max(petal_width, max);

        getline(ss, field, ',');
        T speciesFloat;

        if (field == "Iris-setosa") {
            speciesFloat = 1.0f;

        } else if (field == "Iris-versicolor") {
            speciesFloat = 2.0f;

        } else if (field == "Iris-virginica") {
            speciesFloat = 3.0f;
        }

        std::vector<T> lineData = {sepal_length, sepal_width, petal_length, petal_width, speciesFloat};
        AllData.emplace_back(lineData);

    }

    
    file.close();

    for (const auto& vect : AllData) {

        IrisTuple data;
        std::tuple<int, int, int> species_tuple;

        auto normalizedSL = normalize(vect[0], min, max);
        auto normalizedSW = normalize(vect[1], min, max);
        auto normalizedPL = normalize(vect[2], min, max);
        auto normalizedPW = normalize(vect[3], min, max);
        auto code = vect[4];

        
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
template std::tuple<std::vector<std::tuple<float, float, float, float, std::tuple<int, int, int>>>, std::vector<std::tuple<float, float, float, float, std::tuple<int, int, int>>>, std::vector<std::tuple<float, float, float, float, std::tuple<int, int, int>>>> readIrisData(const std::string& file_name);
template std::tuple<std::vector<std::tuple<double, double, double, double, std::tuple<int, int, int>>>, std::vector<std::tuple<double, double, double, double, std::tuple<int, int, int>>>, std::vector<std::tuple<double, double, double, double, std::tuple<int, int, int>>>> readIrisData(const std::string& file_name);


template<typename T>
//using IrisTuple = std::tuple<T, T, T, T, std::tuple<int, int, int>>;
std::tuple<std::vector<std::vector<T>>, std::vector<std::vector<T>>,
           std::vector<std::vector<T>>, std::vector<std::vector<T>>,
           std::vector<std::vector<T>>, std::vector<std::vector<T>>>
getIrisSets(const std::tuple<std::vector<std::tuple<T, T, T, T, std::tuple<int, int, int>>>, std::vector<std::tuple<T, T, T, T, std::tuple<int, int, int>>>, std::vector<std::tuple<T, T, T, T, std::tuple<int, int, int>>>>& iris_data,
          double train_ratio, double val_ratio, double test_ratio) {

            using IrisTuple = std::tuple<T, T, T, T, std::tuple<int, int, int>>;

    // Check that the ratios are correct
    double total_ratio = train_ratio + val_ratio + test_ratio;

    if (total_ratio != 1.0) {
        std::cerr << "Error: total ratio must be 1.0." << std::endl;
        exit(1);
    }

    // Create training, validation, and test sets
    std::vector<std::vector<T>> train_set, train_out, val_set, val_out, test_set, test_out;

    auto iris_array = {std::get<0>(iris_data), std::get<1>(iris_data), std::get<2>(iris_data)};

    // Loop over each vector in the iris_array
    for (const auto& species_data : iris_array) {

        // Calculate the number of elements for each set
        std::size_t total_size = species_data.size();
        std::size_t train_size = static_cast<std::size_t>(train_ratio * total_size);
        std::size_t val_size = static_cast<std::size_t>(val_ratio * total_size);
        std::size_t test_size = total_size - train_size - val_size;

        for (size_t i = 0; i < total_size; ++i) {
            std::vector<T> inputSide = {std::get<0>(species_data[i]), std::get<1>(species_data[i]), std::get<2>(species_data[i]), std::get<3>(species_data[i])};
            std::vector<T> outputSide = {static_cast<T>(std::get<0>(std::get<4>(species_data[i]))),
                                             static_cast<T>(std::get<1>(std::get<4>(species_data[i]))),
                                             static_cast<T>(std::get<2>(std::get<4>(species_data[i])))};

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

template std::tuple<std::vector<std::vector<float>>, std::vector<std::vector<float>>,
           std::vector<std::vector<float>>, std::vector<std::vector<float>>,
           std::vector<std::vector<float>>, std::vector<std::vector<float>>>
getIrisSets(const std::tuple<std::vector<std::tuple<float, float, float, float, std::tuple<int, int, int>>>, std::vector<std::tuple<float, float, float, float, std::tuple<int, int, int>>>, std::vector<std::tuple<float, float, float, float, std::tuple<int, int, int>>>>& iris_data,
          double train_ratio, double val_ratio, double test_ratio);


template std::tuple<std::vector<std::vector<double>>, std::vector<std::vector<double>>,
           std::vector<std::vector<double>>, std::vector<std::vector<double>>,
           std::vector<std::vector<double>>, std::vector<std::vector<double>>>
getIrisSets(const std::tuple<std::vector<std::tuple<double, double, double, double, std::tuple<int, int, int>>>, std::vector<std::tuple<double, double, double, double, std::tuple<int, int, int>>>, std::vector<std::tuple<double, double, double, double, std::tuple<int, int, int>>>>& iris_data,
          double train_ratio, double val_ratio, double test_ratio);


