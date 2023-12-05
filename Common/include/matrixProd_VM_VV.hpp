#include "matrix_VM_VV.hpp"
#include<chrono>
#ifndef MATRIXPROD_VM_VV_H
#define MATRIXPROD_VM_VV_H



//************************************************

//Take as input two Matrix saved as Matrix class, plus a reference to a int64_t and returns
//the product of the two matrix plus the time spent for the function

//***********************************************

template<typename T>
Matrix<T> matrixProd(Matrix<T>& a, Matrix<T>& b, int64_t& dt_01){
  const auto t0 = std::chrono::high_resolution_clock::now();
  Matrix<T> c;
  if(a.ncols() != b.nrows()){
    std::cout << "matrici non moltiplicabili: errore nel numero righe-colonne" << std::endl;
    return c;
  }
  for(size_t i = 0; i < a.nrows(); i++){
    for(size_t j = 0; j < b.ncols(); j++){
      for(size_t r = 0; r < a.ncols(); r++){
        c(i,j) += a(i,r)*b(r,j);
      }
    }
  }
  const auto t1 = std::chrono::high_resolution_clock::now();
  dt_01 = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
  return c;
}

//************************************************

//Take as input two Matrix saved as MatrixVect class, plus a reference to a int64_t and returns
//the product of the two matrix plus the time spent for the function

//***********************************************

template<typename T>
MatrixVect<T> matrixProdVect(MatrixVect<T>& a, MatrixVect<T>& b, int64_t& dt_01){
  const auto t0 = std::chrono::high_resolution_clock::now();
  MatrixVect<T> c;
  if(a.ncols() != b.nrows()){
    std::cout << "matrici non moltiplicabili: errore nel numero righe-colonne" << std::endl;
    return c;
  }
  for(size_t i = 0; i < a.nrows(); i++){
    for(size_t j = 0; j < b.ncols(); j++){
      for(size_t r = 0; r < a.ncols(); r++){
        c(i,j) += a(i,r)*b(r,j);
      }
    }
  }
  const auto t1 = std::chrono::high_resolution_clock::now();
  dt_01 = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
  return c;
}


//************************************************

//Take as input 3 Matrix saved as one dimensional std::vector: a mxq, b qxn, and a reference to an empty
//std::vector c where the function will store the result of the product a*b, plus a reference to a int64_t that returns
// the time spent for the function

//***********************************************

template<typename T>
int MatrixNaive(std::vector<T>& a, std::vector<T>& b, std::vector<T>& c, size_t m, size_t n,  size_t nb, int64_t& dt_01){
  const auto t0 = std::chrono::high_resolution_clock::now();
  for (size_t row = 0; row < m; row++) {
    for (size_t col = 0; col < nb; col++) {
      for (size_t inner = 0; inner < n; inner++) {
        c[row * nb + col] +=
            a[row * n + inner] * b[inner * nb + col];
} } }
  const auto t1 = std::chrono::high_resolution_clock::now();
  dt_01 = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
  return 301;
}

//************************************************

//Take as input 3 Matrix saved as one dimensional std::vector: a mxq, b qxn, and a reference to an empty
//std::vector c where the function will store the result of the product a*b, plus a reference to a int64_t that returns
// the time spent for the function
//the partial result as optimisation is storedd in a register during the for loop before storing the final result in c

//***********************************************

template<typename T>
int MatrixRegOptimised(std::vector<T>& a, std::vector<T>& b, std::vector<T>& c, size_t m, size_t n,  size_t nb, int64_t& dt_01){
  const auto t0 = std::chrono::high_resolution_clock::now();
  for (size_t row = 0; row < m; row++) {
    for (size_t col = 0; col < nb; col++) {
      float acc = 0.0;
      for (size_t inner = 0; inner < n; inner++) {
        acc += a[row * n + inner] * b[inner * nb + col];
      }
      c[row * nb + col] = acc;
} } 
  const auto t1 = std::chrono::high_resolution_clock::now();
  dt_01 = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
  return 302;
}

//************************************************

//Take as input 3 Matrix saved as one dimensional std::vector: a mxq, b qxn, and a reference to an empty
//std::vector c where the function will store the result of the product a*b, plus a reference to a int64_t that returns
// the time spent for the function
//the orde of the for loop is chanched tring to optimise the cache misses

//***********************************************

template<typename T>
int MatrixCaheOptimised(std::vector<T>& a, std::vector<T>& b, std::vector<T>& c, size_t m, size_t n,  size_t nb, int64_t& dt_01){
  const auto t0 = std::chrono::high_resolution_clock::now();
  for (size_t row = 0; row < m; row++) {
    for (size_t inner = 0; inner < n; inner++) {
      for (size_t col = 0; col < nb; col++) {
        c[row * nb + col] +=
            a[row * n + inner] * b[inner * nb + col];
} } } 
  const auto t1 = std::chrono::high_resolution_clock::now();
  dt_01 = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
  return 303;
}

//************************************************

//Take as input 3 Matrix saved as one dimensional std::vector: a mxq, the transpose of b qxn, and a reference to an empty
//std::vector c where the function will store the result of the product a*b, plus a reference to a int64_t that returns
// the time spent for the function
//b is passed as transpose tring to optimise the cache misses

//***********************************************


template<typename T>
int MatrixBTransposeOptimised(std::vector<T>& a, std::vector<T>& b_transpose, std::vector<T>& c, size_t m, size_t n,  size_t nb, int64_t& dt_01){
  const auto t0 = std::chrono::high_resolution_clock::now();
  for (size_t row = 0; row < m; row++) {
    for (size_t col = 0; col < nb; col++) {
      for (size_t inner = 0; inner < n; inner++) {
        c[row * nb + col] +=
            a[row * n + inner] * b_transpose[col * n + inner];
} } } 
  const auto t1 = std::chrono::high_resolution_clock::now();
  dt_01 = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
  return 304;
}

//*********************************************************************

//this function take a Matrix mxn savede in a one dimensions std::vector and his and return his transpose

//**********************************************************************


template<typename T>
std::vector<T> MatrixTranspose(std::vector<T>& a, size_t m, size_t n){
  std::vector<T> a_transpose;
  a_transpose.resize(m*n);
  for(size_t i = 0; i<m; i++){
    for(size_t j = 0; j<n; j++){
      a_transpose[j*m+i] = a[i*n+j];
    }
  }
  return a_transpose;
}



#endif