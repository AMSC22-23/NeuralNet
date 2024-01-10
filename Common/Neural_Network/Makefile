OPTIMIZATION_FLAGS = -std=c++20 -O3 -march=native -ffast-math
AVX_FLAGS = -mavx2 -mfma


NeuralNet:  amsc_nnet.cpp ../src/irisLoader.cpp ../src/network_functions.cpp ../src/ActivationFunctions.cpp  ../src/matrixProd_AVX.cpp
	@echo "Compile and linking..."
	@g++ ${OPTIMIZATION_FLAGS} ${AVX_FLAGS}  -I ../include  amsc_nnet.cpp ../src/irisLoader.cpp ../src/network_functions.cpp ../src/ActivationFunctions.cpp  ../src/matrixProd_AVX.cpp  -o amsc_nnet
	@echo "Done! To execute the neural network: ./amsc_nnet"


