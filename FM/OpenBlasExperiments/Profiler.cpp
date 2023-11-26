#include "Profiler.hpp"
#include<iostream>



const int64_t Profiler::mmm_blas(Matrix<double>& A, Matrix<double>& B){

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



void Profiler::profile(){

    std::cout<<"Profiling with dimensions: "<<std::endl; 
    for (std::size_t dim : dimensions)
        std::cout<<dim<<", "; 

    std::cout<<std::endl; 
    

    for (std::size_t dim : dimensions)
        {
            std::cout<<"Starting profiling with n = " << dim <<std::endl; 


                Matrix<double> A(dim, dim);
                A.random_fill(-10, 10); 

                Matrix<double> B(dim, dim); 
                B.random_fill(-10, 10);

                auto dt = mmm_blas(A, B);

                std::cout<<"Elapsed time [ms]: "<<dt<<std::endl; 
        }


    
  



}; 

