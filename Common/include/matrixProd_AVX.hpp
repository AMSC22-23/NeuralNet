#ifndef MATRIXPROD_AVX_H
#define MATRIXPROD_AVX_H

#include <immintrin.h> //intrinsics intel per SIMD
#include <vector>


////************************************************

//Take as input 3 Matrix saved as one dimensional std::vector: a mxq, b qxn, and a reference to an empty
//std::vector c where the function will store the result of the product a*b, plus a reference to a int64_t that returns
// the time spent for the function
//the function is built with AVX sintax in order to exploit vector registers

//***********************************************


template<typename T>
int matrixMult_Avx(const std::vector<T>& a, const std::vector<T>& b,std::vector<T>& c, size_t ma, size_t na, size_t nb, int64_t& dt_01);

////************************************************

//Take as input 3 Matrix saved as one dimensional std::vector: a mxq,the transpose of b qxn, and a reference to an empty
//std::vector c where the function will store the result of the product a*b, plus a reference to a int64_t that returns
// the time spent for the function
//the function is built with AVX sintax in order to exploit vector registers and b is taken as transpose in order to exploit 
//different avx functions and a better cache esage

//***********************************************

template<typename T>
//int matrixMultTransposeOpt_Avx(std::vector<T>& a, std::vector<T>& b_transpose, std::vector<T>& c, size_t m,int64_t& dt_01);
int matrixMultTransposeOpt_Avx(std::vector<T>& a, std::vector<T>& b_transpose, std::vector<T>& c, size_t ma, size_t na, size_t nb, int64_t& dt_01);





#endif