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

### NEURAL NETWORK
In the context of neural network implementation:

The network.hpp file comprises classes for handling input, output, and layers. The Input class manages training, validation, and test data, allowing easy retrieval and shaping of input sets. The Output class handles target data for training, with activation functions for output layers. The Layer class represents a neural network layer, specifying its name, neuron count, and activation function.

Moving to model.hpp:
The Model class integrates Input, Output, and Layer classes to create a flexible neural network model. It supports dynamic layer addition and provides insights into the model's structure. Key methods include model construction, weight initialization, and execution of predictions and training.

In asmc_nnet.cpp:
Utility functions like genFakeData, normalize, readIrisData, and getIrisSets facilitate dataset generation, normalization, and Iris dataset handling. The main function reads and splits the Iris dataset, creates a customizable neural network model, builds and trains the model using the provided datasets.

It is possible to compile the code with:

```bash
g++ -O3 -std=c++17  -march=native -ffast-math amsc_nnet.cpp ../include/network_functions.cpp ../src/ActivationFunctions.cpp  -std=c++20 -o amsc_nnet
```
