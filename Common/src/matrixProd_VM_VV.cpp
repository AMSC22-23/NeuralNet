#include "../include/matrixProd_VM_VV.hpp"
#include "../include/matrix_VM_VV.hpp"

template<typename T>
Matrix<T> matrixProd(Matrix<T>& a, Matrix<T>& b){
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
  return c;
}

template<typename T>
MatrixVect<T> matrixProdVect(MatrixVect<T>& a, MatrixVect<T>& b){
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
  return c;
}
