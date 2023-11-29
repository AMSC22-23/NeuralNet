#include "Profiler.hpp"
#include<iostream>

#include <fstream>

const int64_t Profiler::mmm_blas(Matrix<double>& A, Matrix<double>& B){
    

   // Double version!
    std::size_t m, n, k; 

    m = A.get_rows(); 
    n = B.get_cols(); 
    k = A.get_cols(); 



    std::vector<double> C(m*n);

    const auto t0 = std::chrono::high_resolution_clock::now();

    cblas_dgemm(
                CblasRowMajor,      // Specifies row-major (C) or column-major (Fortran) data ordering.
                CblasNoTrans,       // Specifies whether to transpose matrix A.
                CblasNoTrans,       // Specifies whether to transpose matrix B.
                m,             // Number of rows in matrices A and C.
                n,             // Number of columns in matrices B and C.
                k,             // Number of columns in matrix A; number of rows in matrix B.
                1.0,                // Scaling factor for the product of matrices A and B.
                A.get_ptr(),       // UnsafePointer<Double>! to Matrix A.
                k,                  
                B.get_ptr(),     // Matrix B. 
                n,             // The size of the first dimension of matrix B; if you are passing a matrix B[m][n], the value should be m.
                0.0,                // Scaling factor for matrix C.
                C.data(),     // Matrix C.
                n             // The size of the first dimension of matrix C; if you are passing a matrix C[m][n], the value should be m.
                );


    const auto t1 = std::chrono::high_resolution_clock::now();

    const auto dt = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
    Matrix<double> result (m, n, C); 

    

    //return result; 

    return dt;
}; 

const int64_t Profiler::mmm_blas(Matrix<float>& A, Matrix<float>& B){

    // Float version!
    std::size_t m, n, k; 

    m = A.get_rows(); 
    n = B.get_cols(); 
    k = A.get_cols(); 



    std::vector<float> C(m*n);

    const auto t0 = std::chrono::high_resolution_clock::now();

    cblas_sgemm(
                CblasRowMajor,      // Specifies row-major (C) or column-major (Fortran) data ordering.
                CblasNoTrans,       // Specifies whether to transpose matrix A.
                CblasNoTrans,       // Specifies whether to transpose matrix B.
                m,             // Number of rows in matrices A and C.
                n,             // Number of columns in matrices B and C.
                k,             // Number of columns in matrix A; number of rows in matrix B.
                1.0,                // Scaling factor for the product of matrices A and B.
                A.get_ptr(),       // UnsafePointer<Double>! to Matrix A.
                k,                  
                B.get_ptr(),     // Matrix B. 
                n,             // The size of the first dimension of matrix B; if you are passing a matrix B[m][n], the value should be m.
                0.0,                // Scaling factor for matrix C.
                C.data(),     // Matrix C.
                n             // The size of the first dimension of matrix C; if you are passing a matrix C[m][n], the value should be m.
                );


    const auto t1 = std::chrono::high_resolution_clock::now();

    const auto dt = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
    Matrix<float> result (m, n, C); 

    

    //return result; 

    return dt;
}; 

const int64_t Profiler::mmm_naive(Matrix<double>& A, Matrix<double>& B){

std::size_t rows = A.get_rows(), columns = B.get_cols(), inners = A.get_cols(); 
std::vector<double> result(rows*columns);

const auto t0 = std::chrono::high_resolution_clock::now();

  for (std::size_t row = 0; row < rows; row++) {
    for (std::size_t col = 0; col < columns; col++) {
      for (std::size_t inner = 0; inner < inners; inner++) {
        result[row * columns + col] +=
            A[row * columns + inner] * B[inner * columns + col];
        } 
    } 
} 

const auto t1 = std::chrono::high_resolution_clock::now();
const auto dt = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();

// Matrix<double> C(rows, columns, result); 




return dt; 
} 

const int64_t Profiler::mmm_naive(Matrix<float>& A, Matrix<float>& B){

std::size_t rows = A.get_rows(), columns = B.get_cols(), inners = A.get_cols(); 
std::vector<float> result(rows*columns);

const auto t0 = std::chrono::high_resolution_clock::now();

  for (int row = 0; row < rows; row++) {
    for (int col = 0; col < columns; col++) {
      for (int inner = 0; inner < inners; inner++) {
        result[row * columns + col] +=
            A[row * columns + inner] * B[inner * columns + col];
        } 
    } 
} 

const auto t1 = std::chrono::high_resolution_clock::now();
const auto dt = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();

// Matrix<float> C(rows, columns, result); 




return dt; 
} 






void Profiler::profile(){

    std::cout<<"Profiling with dimensions: "<<std::endl; 
    for (std::size_t dim : dimensions)
        std::cout<<dim<<", "; 

    std::cout<<std::endl; 
    
    

    std::fstream fout;
    fout.open(outputfile, std::ios::out | std::ios::app); 

    fout<<"AlgorithmID,n,datatype,time"<<std::endl; 

    for (std::size_t dim : dimensions)
        {
            


                Matrix<double> A(dim, dim);
                Matrix<double> B(dim, dim); 
                Matrix<float> Af(dim, dim); 
                Matrix<float> Bf(dim, dim); 


                A.random_fill(-10, 10); 
                B.random_fill(-10, 10);
                Af.random_fill(-10, 10); 
                Bf.random_fill(-10, 10);
                

                auto dt_double = mmm_blas(A, B);
                //auto dt_float = mmm_blas(Af, Bf); 
                auto dt_naive_double = mmm_naive(A, B);

                
                fout<<"blas,"<<dim<<",double,"<<dt_double<<std::endl; 
                //fout<<"blas,"<<dim<<",float,"<<dt_float<<std::endl;
                fout<<"naive,"<<dim<<",double,"<<dt_naive_double<<std::endl; 
        }



    
    fout.close(); 



}; 

