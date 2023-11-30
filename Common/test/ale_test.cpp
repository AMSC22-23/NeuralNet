///test for the single functions
///Ale 29-11-2023

//#include "../include/MatrixSkltn.hpp"
#include "../include/matrix_VM_VV.hpp"
#include "../include/matrixProd_VM_VV.hpp"
#include "../include/matrixProd_AVX.hpp"
#include "../include/MatrixFlat.hpp"



//#include <iostream>   //ok in matrixskltn
//#include <vector>     //ok in matrixskltn
//#include <set>
//#include <algorithm>
//#include <random>       //ok in matrixskltn
#include <chrono>   //per le funzioni di timing

//#include <unordered_map>
//#include <tuple>
//#include <memory>
//#include <cassert>
//#include <utility>
//#include <functional>
//#include <array>

//#include <fstream>  //per output su file
//#include <immintrin.h> //intrinsics intel per SIMD







int main(){

    Matrix<double> a,b,c;
    MatrixVect<double> d,e,f;
    MatrixFlat<double> mat(8,8,0.0,10.0);
    int64_t t1,t2,t3;
    std::vector<double> res,mat1;


    a(0,0) = d(0,0) = 2.0;
    b(0,0) = e(0,0) = 3.0;
    c = matrixProd<double>(a,b,t1);
    f = matrixProdVect<double>(d,e,t2);

    std::cout << "primo: " << c(0,0) <<" secondo: " << f(0,0) << std::endl;
    std::cout << "primo: " << t1 <<" ms secondo: " << t2 << std::endl;
    std::cout << "stampa dati random matrice: " << std::endl;

    for(int i=0; i<mat.getMdata().size(); i++){
        std::cout << mat.getMdata()[i] << " ";
    }
    std::cout << "fine stampa dati random matrice: " << mat.getMdata().size() << std::endl;

    res.resize(8*8);
    mat1=mat.getMdata();
    matrixMult_Avx(mat1,mat1, res, 8,t3);

    for(int i=0; i<mat.getMdata().size(); i++){
        std::cout << res[i] << " ";
    }






    std::cout << "Funziona !!" <<std::endl;
    return 0;
}

