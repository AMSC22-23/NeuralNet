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
#include "cuda_launcher.h"



void printFile(std::ofstream& file, int id, size_t m, size_t n, int T, int64_t& time){
    if(T==0){
        file << "AC;" << id << ";" << m << "x" << n << ";" << "float;" << time << std::endl; 
    }else{
        file << "AC;" << id << ";" << m << "x" << n << ";" << "double;" << time << std::endl;
    }

}

template<typename T>
void printMatrixToFile(int id, std::ofstream& outputFile, std::vector<T>& a, size_t m, size_t n){
    size_t y=33;
    outputFile << "output results of function: " << id << std::endl;
    if(m<y && n<y){
        for(size_t i = 0; i<m; i++){
            for(size_t j =0; j<n; j++){
                outputFile << a[i*n + j] << " ";
            }
            outputFile << std::endl;
        }
    }
    outputFile << std::endl;
}

template<typename T>
void generateStandardMatrix(std::vector<T>& a, size_t m, size_t n){
    for(size_t i = 0; i<m; i++){
            for(size_t j =0; j<n; j++){
                if( i == j){
                    a[i*n + j] = 2;
                }else{
                     a[i*n + j] = 1;
                }
            }
        }
}
/**
void proveCuda(float *a, float *b, float *c, int m){
    std::cout << "vediamo se funziona " << a[0] << " " << b[0] <<std::endl;
    cudaMallocManaged((void **) &a, sizeof(float)*m);
    cudaMallocManaged((void **) &b, sizeof(float)*m);
    cudaMallocManaged((void **) &c, sizeof(float)*m);

    std::cout << "salvate in teoria..." << std::endl;

    cudaFree(a);
    cudaFree(b);
    cudaFree(c);

}**/



int main(){
using namespace std::chrono;
    std::ofstream outputFile("AleResuls.csv", std::ios::app);
    std::ofstream outputMatrixFile("AleMAtrixResuls.csv");
    Matrix<double> z;
    MatrixVect<double> zv;
    int mul=5;
    int m=1000*mul, n=1000*mul,mb=1000*mul,nb=1000*mul, comp_opt = 10*3;

    std::cout << "Building matricies..." << std::endl;
    MatrixFlat<double> mat(m,n,0.0,10.0), matB(mb,nb,0.0,10.0);
    MatrixFlat<float> matf(m,n,0.0,10.0),matfB(mb,nb,0.0,10.0);
    int64_t t1;
    std::vector<double> res,mat1, mat1_transpose,resAVX,mat1AVX, mat1_transposeAVX,mat1B,mat1BAVX,mat1B_transposeAVX,mat1B_transpose;
    std::vector<float> resf,mat1f, mat1f_transpose,resfAVX,mat1fAVX, mat1f_transposeAVX,mat1Bf,mat1BfAVX,mat1Bf_transposeAVX,mat1Bf_transpose;
    int id, i=0,d=0,ib=0,db=0;

    //aggiorno le dimensioni delle matrici per fare in modo che non creino problemi con i registri AVX
    while((m+i) % 8 != 0){
        i++;
    }
    
    while((n+d) % 8 != 0){
        d++;
    }
    while((mb+ib) % 8 != 0){
        ib++;
    }
    
    while((nb+db) % 8 != 0){
        db++;
    }

std::cout << "valori degli offset: " << i <<d <<ib<<db << std::endl;

    res.resize(m*n);
    resf.resize(m*n);
    mat1=mat.getMdata();
    mat1B=matB.getMdata();
    mat1f=matf.getMdata();
    mat1Bf=matfB.getMdata();
    mat1AVX.resize((m+i)*(n+d));
    mat1BAVX.resize((mb+ib)*(nb+db));
    mat1fAVX.resize((m+i)*(n+d));
    mat1BfAVX.resize((mb+ib)*(nb+db));

    //generateStandardMatrix(mat1, m, n);
    //generateStandardMatrix(mat1f, m, n);
    //generateStandardMatrix(mat1Bf, mb, nb);
    //generateStandardMatrix(mat1B, mb, nb);

    //costruisco i puntatore per cuda***************
    float *a,*b,*c;
    a = mat1f.data();
    b = mat1Bf.data();
    c = resf.data();


    //cudaFunction(a,b,c,m,n,nb);
    //cudaLauncher(a, b, c, m, n, nb);
/**
    for(int r=0; r<20 ; r++){
        for(int rr=0; rr<20; rr++){
            std::cout << c[rr+r*nb] << " ";
        }
        std::cout << std::endl;
    }**/


    //printMatrixToFile(00, outputMatrixFile, mat1Bf, mb ,nb);

    //inserisco 0 per rendere righe e colonne multipli di 8 e non creare problemi con i registri AVX
    for(int y =0 ; y<m+i; y++){
        for(int u =0; u<n+d; u++){
            if(u<n && y<m){
                mat1AVX[y*(n+d)+u] = mat1[y*n+u];
                mat1fAVX[y*(n+d)+u] = mat1f[y*n+u];
            }else{
                mat1AVX[y*(n+d)+u] = 0;
                mat1fAVX[y*(n+d)+u] = 0;
            }

        }
    }
    for(int y =0 ; y<mb+ib; y++){
        for(int u =0; u<nb+db; u++){
            if(u<nb && y<mb){
                mat1BfAVX[y*(nb+db)+u] = mat1Bf[y*nb+u];
                mat1BAVX[y*(nb+db)+u] = mat1B[y*nb+u];

            }else{
                mat1BfAVX[y*(nb+db)+u] = 0;
                mat1BAVX[y*(nb+db)+u] = 0;
            }

        }
    }



    mat1_transpose = MatrixTranspose(mat1, m, n);
    mat1f_transpose = MatrixTranspose(mat1f, m, n);
    mat1_transposeAVX = MatrixTranspose(mat1AVX, m+i, n+d);
    mat1f_transposeAVX = MatrixTranspose(mat1fAVX, m+i, n+d);
    mat1Bf_transpose = MatrixTranspose(mat1Bf, mb, nb);
    mat1Bf_transposeAVX = MatrixTranspose(mat1BfAVX, mb+ib, nb+db);
    mat1B_transpose = MatrixTranspose(mat1B, mb, nb);
    mat1B_transposeAVX = MatrixTranspose(mat1BAVX, mb+ib, nb+db);

    //printMatrixToFile(1, outputMatrixFile, mat1, m ,n);
    //printMatrixToFile(2, outputMatrixFile, mat1fAVX, m+i ,n+d);
    //printMatrixToFile(11, outputMatrixFile, mat1f_transposeAVX, n+d ,m+i);
    //printMatrixToFile(20, outputMatrixFile, mat1BfAVX, mb+ib ,nb+db);
    //printMatrixToFile(22, outputMatrixFile, mat1Bf_transposeAVX, nb+db ,mb+ib);

    
    
    std::cout << "Test starting..." << std::endl;
    if(n != mb){
        std::cout << "Matrici A e B non moltiplicabili: errore nelle dimensioni" << std::endl;
        return 0;

    }

    cudaFunction(a,b,c,m,n,nb);

/**
    res.resize(m*nb);
    res.assign(m*nb,0);
    //id = MatrixNaive(mat1, mat1B, res, m,n,nb, t1);
    std::cout<<std::endl;
    //std::cout << "funzione: " << id << " tempo: " << t1 << " ms" << std::endl;
    //printFile(outputFile,id+comp_opt,m,nb,1,t1);
    std::cout<<std::endl;
    //printMatrixToFile(id, outputMatrixFile, res, m, nb);
   
    res.resize(m*nb);
    res.assign(m*nb,0);
    //id = MatrixRegOptimised(mat1, mat1B, res, m,n,nb, t1);
    std::cout<<std::endl;
    //std::cout << "funzione: " << id << " tempo: " << t1 << " ms" << std::endl;
    //printFile(outputFile,id+comp_opt,m,nb,1,t1);
    std::cout<<std::endl;
    //printMatrixToFile(id, outputMatrixFile, res, m, nb);
    
    res.resize(m*nb);
    res.assign(m*nb,0);
    id = MatrixCaheOptimised(mat1, mat1B, res, m,n,nb, t1);
    std::cout<<std::endl;
    std::cout << "funzione: " << id << " tempo: " << t1 << " ms" << std::endl;
    printFile(outputFile,id+comp_opt,m,nb,1,t1);
    std::cout<<std::endl;
    //printMatrixToFile(id, outputMatrixFile, res, m, nb);
    res.assign(m*n, 0);

    res.resize(m*nb);
    res.assign(m*nb,0);
    //id = MatrixBTransposeOptimised(mat1, mat1B_transpose, res, m,n,nb, t1);
    std::cout<<std::endl;
    //std::cout << "funzione: " << id << " tempo: " << t1 << " ms" << std::endl;
    //printFile(outputFile,id+comp_opt,m,nb,1,t1);
    std::cout<<std::endl;
    //printMatrixToFile(id, outputMatrixFile, res, m, nb);
    
    res.resize((m+i)*(nb+db));
    res.assign((m+i)*(nb+db), 0);
    //id = matrixMult_Avx(mat1AVX, mat1BAVX, res, m+i,n+d,nb+db, t1);
    std::cout<<std::endl;
    //std::cout << "funzione: " << id << " tempo: " << t1 << " ms" << std::endl;
    //printFile(outputFile,id+comp_opt,m+i,nb+db,1,t1);
    std::cout<<std::endl;
    //printMatrixToFile(id, outputMatrixFile, res, m+i, nb+db);
   
    res.assign((m+i)*(nb+db), 0);
    id = matrixMultTransposeOpt_Avx(mat1AVX, mat1B_transposeAVX, res, m+i,n+d,nb+db, t1);
    std::cout<<std::endl;
    std::cout << "funzione: " << id << " tempo: " << t1 << " ms" << std::endl;
    printFile(outputFile,id+comp_opt,m+i,nb+db,1,t1);
    std::cout<<std::endl;
    //printMatrixToFile(id, outputMatrixFile, res, m+i, nb+db);
    


    std::cout << "***********************************************" << std::endl;
    
    resf.resize(m*nb);
    resf.assign(m*nb,0);
    //id = MatrixNaive(mat1f, mat1Bf, resf, m,n,nb, t1);
    std::cout<<std::endl;
    //std::cout << "funzione: " << id << " tempo: " << t1 << " ms" << std::endl;
    //printFile(outputFile,id+comp_opt,m,nb,0,t1);
    std::cout<<std::endl;
    //printMatrixToFile(id, outputMatrixFile, resf, m, nb);
    
    resf.assign(m*nb,0);
    //id = MatrixRegOptimised(mat1f, mat1Bf, resf, m,n,nb, t1);
    std::cout<<std::endl;
    //std::cout << "funzione: " << id << " tempo: " << t1 << " ms" << std::endl;
    //printFile(outputFile,id+comp_opt,m,nb,0,t1);
    std::cout<<std::endl;
    //printMatrixToFile(id, outputMatrixFile, resf, m, nb);
    

    resf.resize(m*nb);
    resf.assign(m*nb, 0);
    id = MatrixCaheOptimised(mat1f, mat1Bf, resf, m,n,nb, t1);
    std::cout<<std::endl;
    std::cout << "funzione: " << id << " tempo: " << t1 << " ms" << std::endl;
    printFile(outputFile,id+comp_opt,m,nb,0,t1);
    std::cout<<std::endl;
    //printMatrixToFile(id, outputMatrixFile, resf, m, nb);
    
    resf.assign(m*nb, 0);
    //id = MatrixBTransposeOptimised(mat1f, mat1Bf_transpose, resf, m,n,nb, t1);
    std::cout<<std::endl;
    //std::cout << "funzione: " << id << " tempo: " << t1 << " ms" << std::endl;
    //printFile(outputFile,id+comp_opt,m,nb,0,t1);
    std::cout<<std::endl;
    //printMatrixToFile(id, outputMatrixFile, resf, m, nb);
    



    resf.resize((m+i)*(nb+db));
    resf.assign((m+i)*(nb+db), 0);
    //id = matrixMult_Avx(mat1fAVX, mat1BfAVX, resf, m+i,n+d,nb+db, t1);
    std::cout<<std::endl;
    //std::cout << "funzione: " << id << " tempo: " << t1 << " ms" << std::endl;
    //printFile(outputFile,id+comp_opt,m+i,nb+db,0,t1);
    std::cout<<std::endl;
    //printMatrixToFile(id, outputMatrixFile, resf, m+i, nb+db);
    



    resf.resize((m+i)*(nb+db));
    resf.assign((m+i)*(nb+db), 0);
    id = matrixMultTransposeOpt_Avx(mat1fAVX, mat1Bf_transposeAVX, resf, m+i, n+d, nb+db, t1);
    std::cout<<std::endl;
    std::cout << "funzione: " << id << " tempo: " << t1 << " ms" << std::endl;
     printFile(outputFile,id+comp_opt,m+i,nb+db,0,t1);
    std::cout<<std::endl;
    //printMatrixToFile(id, outputMatrixFile, resf, m+i, nb+db);
**/
    /**
    for(int r=0; r<20 ; r++){
        for(int rr=0; rr<20; rr++){
            std::cout << resf[rr+i+r*(nb+db)] << " ";
        }
        std::cout << std::endl;
    }**/


    outputFile.close();
    outputMatrixFile.close();

    std::cout << "Funziona !!" <<std::endl;
    return 0;
}

