#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <tuple>

// Tuple to represent a row of Iris data
// Each tuple contains 4 doubles (SepalLengthCm, SepalWidthCm, PetalLengthCm, PetalWidthCm)
// and a tuple representing the species in one-hot encoding format
using IrisTuple = std::tuple<double, double, double, double, std::tuple<int, int, int>>;

// Function to read Iris data from a CSV file
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
        // Convert field to double and save the value in sepal_length
        double sepal_length = stod(field);

        getline(ss, field, ',');
        double sepal_width = stod(field);

        getline(ss, field, ',');
        double petal_length = stod(field);

        getline(ss, field, ',');
        double petal_width = stod(field);

        getline(ss, field, ',');
        std::tuple<int, int, int> species_tuple;

        // Represent the species in one-hot encoding format
        if (field == "Iris-setosa") {
            species_tuple = std::make_tuple(1, 0, 0);
            IrisTuple data = std::make_tuple(sepal_length, sepal_width, petal_length, petal_width, species_tuple);
            setosa_data.emplace_back(data);
        } else if (field == "Iris-versicolor") {
            species_tuple = std::make_tuple(0, 1, 0);
            IrisTuple data = std::make_tuple(sepal_length, sepal_width, petal_length, petal_width, species_tuple);
            versicolor_data.emplace_back(data);
        } else if (field == "Iris-virginica") {
            species_tuple = std::make_tuple(0, 0, 1);
            IrisTuple data = std::make_tuple(sepal_length, sepal_width, petal_length, petal_width, species_tuple);
            virginica_data.emplace_back(data);
        }
    }

    // Close the file
    file.close();

    // Return the tuple containing the three vectors of tuples
    return std::make_tuple(setosa_data, versicolor_data, virginica_data);
}

// Function to split the data into a training set, validation set, and test set
// Returns a tuple containing three vectors of IrisTuple
std::tuple<std::vector<IrisTuple>, std::vector<IrisTuple>, std::vector<IrisTuple>>
splitData(const std::tuple<std::vector<IrisTuple>, std::vector<IrisTuple>, std::vector<IrisTuple>>& iris_data,
          double train_ratio, double val_ratio, double test_ratio) {

    // Check that the ratios are correct
    double total_ratio = train_ratio + val_ratio + test_ratio;

    if (total_ratio != 1.0) {
        std::cerr << "Error: total ratio must be 1.0." << std::endl;
        exit(1);
    }

    // Create training, validation, and test sets
    std::vector<IrisTuple> train_set, val_set, test_set;

    auto iris_array = {std::get<0>(iris_data), std::get<1>(iris_data), std::get<2>(iris_data)};

    // Loop over each vector in the iris_array
    for (const auto& species_data : iris_array) {

        // Calculate the number of elements for each set
        std::size_t total_size = species_data.size();
        std::size_t train_size = static_cast<std::size_t>(train_ratio * total_size);
        std::size_t val_size = static_cast<std::size_t>(val_ratio * total_size);
        std::size_t test_size = total_size - train_size - val_size;

        // Add elements to the sets
        train_set.insert(train_set.end(), species_data.begin(), species_data.begin() + train_size);
        val_set.insert(val_set.end(), species_data.begin() + train_size, species_data.begin() + train_size + val_size);
        test_set.insert(test_set.end(), species_data.begin() + train_size + val_size, species_data.end());
    }

    // Return the tuple containing the three sets
    return std::make_tuple(train_set, val_set, test_set);
}

int main() {
    // Name of the CSV file to read
    std::string file_name = "Iris.csv";

    // Call the function to get the data
    auto iris_data = readIrisData(file_name);

    // Split the data into a training set, validation set, and test set
    // Insert the percentage of total data to assign to each set
    auto split_result = splitData(iris_data, 0.5, 0.3, 0.2);

    // Now split_result contains the three vectors corresponding to the three sets

    // Example: print the size of each set
    std::cout << "Training Set Size: " << std::get<0>(split_result).size() << std::endl;
    std::cout << "Validation Set Size: " << std::get<1>(split_result).size() << std::endl;
    std::cout << "Test Set Size: " << std::get<2>(split_result).size() << std::endl;

    return 0;
}