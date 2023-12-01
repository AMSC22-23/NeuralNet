///test for the single functions
///Ale 29-11-2023

//#include "../include/MatrixSkltn.hpp"
#include "../include/matrix_VM_VV.hpp"
#include "../include/matrixProd_VM_VV.hpp"
#include "../include/matrixProd_AVX.hpp"
#include "../include/MatrixFlat.hpp"



//#include <iostream>   //ok in matrixskltn
//#include <vector>     //ok in matrixskltn
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



void printFile(std::ofstream& file, int id, size_t m, size_t n, int T, int64_t& time){
    if(T==0){
        file << "AC;" << id << ";" << m << "x" << n << ";" << "float;" << time << std::endl; 
    }else{
        file << "AC;" << id << ";" << m << "x" << n << ";" << "double;" << time << std::endl;
    }

}



int main(){
using namespace std::chrono;
    std::ofstream outputFile("AleResuls.csv", std::ios::app);
    Matrix<double> a,b,c;
    MatrixVect<double> d,e,f;
    int m=1024, n=1024;

    std::cout << "Building matricies..." << std::endl;
    MatrixFlat<double> mat(m,n,0.0,10.0);
    MatrixFlat<float> matf(m,n,0.0,10.0);
    int64_t t1,t2,t3;
    std::vector<double> res,mat1;
    std::vector<float> resf,mat1f;
    int id;


    res.resize(m*n);
    resf.resize(m*n);
    mat1=mat.getMdata();
    mat1f=matf.getMdata();
    
    std::cout << "Test starting..." << std::endl;

    id = MatrixNaive(mat1, mat1, res, m,n, t1);
    std::cout<<std::endl;
    std::cout << "funzione: " << id << " tempo: " << t1 << " ms" << std::endl;
    printFile(outputFile,id,m,n,1,t1);
    std::cout<<std::endl;
    
    id = MatrixRegOptimised(mat1, mat1, res, m,n, t1);
    std::cout<<std::endl;
    std::cout << "funzione: " << id << " tempo: " << t1 << " ms" << std::endl;
    printFile(outputFile,id,m,n,1,t1);
    std::cout<<std::endl;

    id = MatrixCaheOptimised(mat1, mat1, res, m,n, t1);
    std::cout<<std::endl;
    std::cout << "funzione: " << id << " tempo: " << t1 << " ms" << std::endl;
    printFile(outputFile,id,m,n,1,t1);
    std::cout<<std::endl;

    id = MatrixBTransposeOptimised(mat1, mat1, res, m,n, t1);
    std::cout<<std::endl;
    std::cout << "funzione: " << id << " tempo: " << t1 << " ms" << std::endl;
    printFile(outputFile,id,m,n,1,t1);
    std::cout<<std::endl;

    id = matrixMult_Avx(mat1, mat1, res, m, t1);
    std::cout<<std::endl;
    std::cout << "funzione: " << id << " tempo: " << t1 << " ms" << std::endl;
    printFile(outputFile,id,m,n,1,t1);
    std::cout<<std::endl;

    id = matrixMultTransposeOpt_Avx(mat1, mat1, res, m, t1);
    std::cout<<std::endl;
    std::cout << "funzione: " << id << " tempo: " << t1 << " ms" << std::endl;
    printFile(outputFile,id,m,n,1,t1);
    std::cout<<std::endl;


    std::cout << "***********************************************" << std::endl;

    id = MatrixNaive(mat1f, mat1f, resf, m,n, t1);
    std::cout<<std::endl;
    std::cout << "funzione: " << id << " tempo: " << t1 << " ms" << std::endl;
    printFile(outputFile,id,m,n,0,t1);
    std::cout<<std::endl;
    
    id = MatrixRegOptimised(mat1f, mat1f, resf, m,n, t1);
    std::cout<<std::endl;
    std::cout << "funzione: " << id << " tempo: " << t1 << " ms" << std::endl;
    printFile(outputFile,id,m,n,0,t1);
    std::cout<<std::endl;

    id = MatrixCaheOptimised(mat1f, mat1f, resf, m,n, t1);
    std::cout<<std::endl;
    std::cout << "funzione: " << id << " tempo: " << t1 << " ms" << std::endl;
    printFile(outputFile,id,m,n,0,t1);
    std::cout<<std::endl;

    id = MatrixBTransposeOptimised(mat1f, mat1f, resf, m,n, t1);
    std::cout<<std::endl;
    std::cout << "funzione: " << id << " tempo: " << t1 << " ms" << std::endl;
    printFile(outputFile,id,m,n,0,t1);
    std::cout<<std::endl;

    id = matrixMult_Avx(mat1f, mat1f, resf, m, t1);
    std::cout<<std::endl;
    std::cout << "funzione: " << id << " tempo: " << t1 << " ms" << std::endl;
    printFile(outputFile,id,m,n,0,t1);
    std::cout<<std::endl;

    id = matrixMultTransposeOpt_Avx(mat1f, mat1f, resf, m, t1);
    std::cout<<std::endl;
    std::cout << "funzione: " << id << " tempo: " << t1 << " ms" << std::endl;
    printFile(outputFile,id,m,n,0,t1);
    std::cout<<std::endl;




    std::cout << "Funziona !!" <<std::endl;
    return 0;
}

