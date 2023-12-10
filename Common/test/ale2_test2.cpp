//#include "../include/matrix_VM_VV.hpp"
//#include "../include/matrixProd_VM_VV.hpp"
//#include "../include/matrixProd_AVX.hpp"
//#include "../include/MatrixFlat.hpp"



#include <iostream>   //ok in matrixskltn
#include <vector>     //ok in matrixskltn
#include <set>
#include <algorithm>
#include <random>       //ok in matrixskltn
#include <chrono>   //per le funzioni di timing

#include <unordered_map>
#include <tuple>
#include <memory>
#include <cassert>
#include <utility>
#include <functional>
#include <array>

#include <fstream>  //per output su file
#include <immintrin.h> //intrinsics intel per SIMD

int main(){

    std::vector<double> a,b,c ;
    __m256d A,B,C,D;
    a.resize(16);
    a.assign(16,1);
    b.resize(16);
    b.assign(16,2);
    c.resize(16);
    c.assign(16,3);

    A = _mm256_loadu_pd(&a[0]);
    B = _mm256_loadu_pd(&b[0]);
    C = _mm256_loadu_pd(&c[0]);

    D = _mm256_fmadd_pd(A, B, C);

    for(int i=0;i<4;i++){
        std::cout << ((double*)&A)[i] << " ";
    }
    std::cout << std::endl;

    for(int i=0;i<4;i++){
        std::cout << ((double*)&B)[i] << " ";
    }
    std::cout << std::endl;

    for(int i=0;i<4;i++){
        std::cout << ((double*)&C)[i] << " ";
    }
    std::cout << std::endl;

    for(int i=0;i<4;i++){
        std::cout << ((double*)&D)[i] << " ";
    }
    std::cout << std::endl;

    return 0;

}