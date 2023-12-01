#ifndef MATRIXPROD_AVX_H
#define MATRIXPROD_AVX_H

#include <immintrin.h> //intrinsics intel per SIMD
#include <vector>

/**
template<>
void matrixMult_impl_Avx(std::vector<double>& a, std::vector<double>& b, std::vector<double>& c, size_t m){
  //prova su double
    __m256d A,B,result;
    for (size_t i =0; i<m; i++){
       //result = _mm256_setzero_pd();
       for(size_t q = 0; q < m; q +=4){
        result = _mm256_setzero_pd();
          for(size_t j=0; j<m; j++){
            A = _mm256_broadcast_sd(&a[j+i*m]);
            B = _mm256_loadu_pd(&b[j*m+q]);
            result =  _mm256_add_pd(result, _mm256_mul_pd(A, B));
          }
          _mm256_storeu_pd(&c[i*m+q], result);
        }
    }
}

template<>
void matrixMult_impl_Avx(std::vector<float>& a, std::vector<float>& b, std::vector<float>& c, size_t m){
  //prova su double
    __m256 A,B,result;
    for (size_t i =0; i<m; i++){
       //result = _mm256_setzero_pd();
       for(size_t q = 0; q < m; q +=8){
        result = _mm256_setzero_ps();
          for(size_t j=0; j<m; j++){
            A = _mm256_broadcast_ss(&a[j+i*m]);
            B = _mm256_loadu_ps(&b[j*m+q]);
            result =  _mm256_add_ps(result, _mm256_mul_ps(A, B));
          }
          _mm256_storeu_ps(&c[i*m+q], result);
        }
    }
}


template<typename T>
void matrixMult_Avx(std::vector<T>& a, std::vector<T>& b, std::vector<T>& c, size_t m){
    matrixMult_impl_Avx(a, b, c, m);

}**/
template<typename T>
int matrixMult_Avx(std::vector<T>& a, std::vector<T>& b, std::vector<T>& c, size_t m,int64_t& dt_01);

template<typename T>
int matrixMultTransposeOpt_Avx(std::vector<T>& a, std::vector<T>& b, std::vector<T>& c, size_t m,int64_t& dt_01);






#endif