#include "matrix_VM_VV.hpp"
#include<chrono>
#ifndef MATRIXPROD_VM_VV_H
#define MATRIXPROD_VM_VV_H




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

#endif