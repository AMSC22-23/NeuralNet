# NEURAL NETS PROJECT

## FOLDER STRUCTURE
All of our work is located in the "Common" folder in this repository, which includes 4 folders:
- Include: Contains all header files used in the project.
- src: Contains the source code for the matrix multiplication algorithms, profiler, and network's activation functions.
- matrix mult: Contains several different implementation of Matrix x Matrix multiplication
- test: Contains all the unt test and profiling done on the different implementations
- Neural Network: inside here the main function that implement the neural network

## DESCRIPTION OF THE WORK
Our work was divided into three main phases:
- Writing code for different versions of matrix-matrix multiplication algorithms.
- Profiling the performances of all the different versions.
- Building the neural network.

### MATRIX MATRIX MULTIPLICATION
We started working on the project by implementing various versions of matrix-matrix multiplication code individually. This allowed us to profile the performances of different approaches and figure out which approach would be better to use in our neural network. We tried different algorithms and data structures:

- Simple-function-like matrix multiplication.

- Defined and implemented classes that stored the matrices' data using different structures, each structure with its own methods  for accessing the matrix and for matrix-matrix multiplication. We explored the following structures in organizing the data:
    - single std::vector<> row or column major.
    - std::vector<std::vector<>>.
    - std::vector<std::map<>>.

Moreover we tried to implement different classes and methods to understando how different call or data organization can affect the time complexity.



- Tested different loop structures to figure out which was the best for cache performance:
    - Naive functions.
    - Naive functions with accumulations.
    - Cache-optimized function that reorders the two inner loops.
    - Transpose a matrix to reach better cache performancies.
    - Tiling.

- Compiled the code with different optimization flags to compare them and eventually exploit loop unrolling and function inlining.
- Exploited SIMD instructions using vectorized AVX functions.
- Exploited CUDA kernels with different optimizations (block dimension, tiling)
- Utilized OpenBlas library to compare our results with its functions.

**********descrizione degli unit test di filippo e come compilarli

Inside the matrix_mult folder, there are two versions of the same code, ale_test.cpp compiled with:

```bash
g++ -O3 -std=c++20  -march=native -ffast-math ale_test.cpp ../src/matrixProd_AVX.cpp  -mavx2 -mfma -std=c++20 -o ale_test
```
It offers the possibility to evaluate the time complexity of different sequential algorithms. The code accepts an m x n matrix, checks the dimensions, and performs different functions exploiting also vectorized instructions through the AVX library. By modifying these two lines, it is possible to test any dimension as needed.

```c
77  int mul=1;
78  int m=1024*mul, n=1024*mul,mb=1024*mul,nb=1024*mul ...
```
The second file, ale_test_cuda.cpp, needs to perform several optimizations exploiting the GPU with functions written in CUDA. A Google Colab notebook, "AMSC_cuda_profiler.ipynb," is provided that allows you to load, compile, run code and functions, after copying the folder to a Google Drive folder.

### PROFILING
To conduct the profiling of our algorithms and examine the impact of various parameters, we developed a compact program designed to automate this process. This program accepts a CSV file (referred to as "profilelist.txt") as input, containing the specified profiling tasks to be executed. Upon execution, the program generates and records the results in distinct CSV files, specifically "filResult.csv" and "profiling_results.csv." Subsequently, when it comes time to analyze the profiling results, we upload the data contained in "profiling_results.csv" to a MongoDB database. Using the Python MongoDB library, we can query and visualize the required data.

## NEURAL NETWORK
We implemented a Feed Forward Neural Network, and matrix multiplications are handled by the functions developed in the first part of the project. The weights are updated through a Gradient Descent optimization implemented in a batch-wise solution.

In the context of neural network implementation:

The network.hpp file comprises classes for handling input, output, and layers. The Input class manages training, validation, and test data, allowing easy retrieval and shaping of input sets. The Output class handles target data for training, with activation functions for output layers. The Layer class represents a neural network layer, specifying its name, neuron count, and activation function.

Moving to model.hpp:
The Model class integrates Input, Output, and Layer classes to create a flexible neural network model. It supports dynamic layer addition and provides insights into the model's structure. Key methods include model construction, weight initialization, and execution of predictions and training.

In asmc_nnet.cpp:
Utility functions like, readIrisData, and getIrisSets facilitate dataset generation, normalization, and Iris dataset handling. The main function reads and splits the Iris dataset, creates a customizable neural network model, builds and trains the model using the provided datasets.

It is possible to compile the code with:

```bash
g++ -O3 -std=c++17  -march=native -ffast-math amsc_nnet.cpp ../src/irisLoader.cpp ../include/network_functions.cpp ../src/ActivationFunctions.cpp  -std=c++20 -o amsc_nnet
```

### HOW IT WORKS

Opening 'amsc_nnet.cpp' allows for the construction of a custom Neural Network. First, it is necessary to implement the Input and Output classes. The code operates by taking as input the dataset organized in std::vector<std::vector<T>>, where T can be either float or double type. The default constructors for the input and output classes take, as parameters, three different std::vector<std::vector<T>> representing the values of the training set, validation set, and test set. In each line of these variables, the different occurrences in the dataset are stored, with columns representing the available inputs for each occurrence. Output Class take as input also the activation function related to the desired output.

The next step is to define each layer through the `Layer` class. In the constructor, you need to add the following parameters:
- Name of the Layer
- Number of neurons
- Activation Function.

Next, you have to define your model by creating a `Model<T>` object using the `Model` class. The required parameters are:
- Name of the model
- Number of epochs
- Batch Size
- Learning rate
- Object of the class `Input`
- Object of the class `Output`
- Stop criteria

Finally, it is necessary to assemble the layers previously defined using the `addLayer()` method, build the entire structure of the model with the `buildModel()` method, and then launch the training with the `train()` method of the `Model` class.

```c++
// main() function in amsc_nnet.cpp

// Toy example
shuffleData(trainSet, trainOut); // function to shuffle the dataset
Input input(trainSet, validationSet, testSet);
Output output(trainOut, validationOut, testOut, "sigmoid");

Layer layer1("test1", 128, "ReLu"), layer2("test2", 200, "ReLu"), layer3("test3", 300, "ReLu");

Model model("myModel", 100, 16, 0.05, "MSE", input, output, "early_stop");

model.addLayer(test1);
model.addLayer(test2);
model.addLayer(test3);

model.buildModel();

model.train(a); // "a" is the integer parameter that defines the chosen optimization

```

#### Input Class
Below is a list of implemented methods for this class.

```c++

// Default constructor
Input(std::vector<std::vector<T>> train, std::vector<std::vector<T>> validation, std::vector<std::vector<T>> test)

// Copy constructor
Input(const Output<T>& copy)

// Print occurrence of different input sets to the standard output
void Input::printShape()

// Return the size of the input
int Input::getShapeInputData()

// Methods to retrieve different data
std::vector<std::vector<T>> Input::getTrain() 
std::vector<std::vector<T>> Input::getTest() 
std::vector<std::vector<T>> Input::getValidation()

// Given a new set as input, it is possible to set new values in the Input class
void Input::setInputSet(std::vector<std::vector<T>> train, std::vector<std::vector<T>> validation, std::vector<std::vector<T>> test,
            std::vector<std::vector<T>>& train_input, std::vector<std::vector<T>>& validation_input,
            std::vector<std::vector<T>>& test_input)


```

#### Output Class

Below is a list of implemented methods for this class.

```c++

// Default constructor
Output(std::vector<std::vector<T>> train_target, std::vector<std::vector<T>> validadion_target,
         std::vector<std::vector<T>> test_target, std::string sel_actFun)

// Copy constructor
Output(const Output<T>& copy)

// Print occurrence of different output sets to the standard output
void Output::printShape()

// Return the size of the output
int Output::getShapeOutputData()

// Methods to retrieve different data
std::vector<std::vector<T>> Output::getOutputTrain() 
std::vector<std::vector<T>> Output::getOutputTest() 
std::vector<std::vector<T>> Output::getOutputValidation()
std::string Output::getOutputAct_fun()

//Given as reference the varable is possible to change the activation function of the object
void Output::set_Act_Fun(std::string selection, std::string& variable)

// Method used to change the each target variable in the class with the provided value
void setTarget(std::vector<std::vector<T>> new_value, std::vector<std::vector<T>>& variable_you_want_to_change)

```

#### Layer Class
Another list of available methods

```c++
//Default constructor
Layer(std::string layer_name, int num_of_neurons, std::string layer_act_funct)

//Print on the standard output all the details related to the layer
void Layer::printLayer()

//How to retrive usefull data
int getNeurons()
std::string getActFun() 
std::string getName()
```

#### Model Class
This class is the engine of the network, and there are several methods capable of building and running training or predictions. Various methods are offered to customize each default setting or parameter. Flexibility is the most important feature of this class; every single method is callable from the main function to retrieve or implement whatever is needed.
Below a list of available usefull methods:

```c++
//Default constructor
Model(std::string name, int epochs, int batch_size, float learning_rate, std::string loss_fun, Input<T> input, Output<T> output, std::string stop_cryteria)

//How to add a layer to the model
void Model::addLayer(Layer layer)

//print to the standard output the collection of layers saved in the model
void Model::printLayers()

//print to the standard output the actual value of the weights matrix
void Model::printWeigts()

//Set the preferred model in weights initialization, is possible to set as parameters the following models
/**
 *  1) "Normal_Distribution"  (Default)
 *  2) "Uniform_Distribution
 *  3) "He"
 *  4) "Xaviere"
 *  5) "debug"
*/
void Model::setWeightdInitialization(const std::string weights_model)

//print to weights.txt all the values contained in each matrix stored in Model at the moment the function is called
void Model::printAllWeightsToFile()

//print on the standard output all the details of the network in keras-style
void Model::printModel()

//defined the parameters this method build all the structure needed for the training
void Model::buildModel()

//perform a prediction given a single vector of input, flag is a generic integer parameters that allow to call this version of the function
//another overloaded function is available without flag parameter but need to be called only inside the train function
void Model::predict(std::vector<T>& input, int& selection, int flag)

//perform a backpropagation step with the chain-rule
void Model::backPropagation(std::vector<T>& input, std::vector<T>& dE_dy, int& selection)

//perform the training of the network, selection is the parameter that allow you to choose the preferred numerical optimization in matrix multiplication
/**
 *  1) 0 - cache optimize
 *  2) 1 - ...
 * **/
void Model::train(int& selection)

```

There are several other methods that are only usefull utilities runned during the training.

#### A real example
Given the Iris dataset already divided in train, validation and test set using the std::vector<std::vector<T>> structure, following a runnable example of the code



