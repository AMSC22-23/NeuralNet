#! /bin/bash

#g++ ale_test.cpp ../src/matrixProd_VM_VV.cpp -O0 -march=native -mavx2   -std=c++20  -g -Wall -Wextra -pedantic -fsanitize=address

#g++ ale_test.cpp ../src/matrixProd_VM_VV.cpp -c

#g++ -o ale_test.o matrixProd_VM_VV.o

g++ -O3 -std=c++20  -march=native -ffast-math ale_test.cpp ../src/matrixProd_AVX.cpp  -mavx2 -mfma -std=c++20 -o ale_test
#g++ -O0 -std=c++20 ale2_test2.cpp   -mavx2 -mfma -std=c++20 -o ale2_test2

#g++ -std=c++20 ale_test.cpp   -mavx2 -std=c++20 -o ale_test