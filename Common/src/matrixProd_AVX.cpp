#include "../include/matrixProd_AVX.hpp"
#include<chrono>




    template<>
    int matrixMult_Avx(std::vector<double>& a, std::vector<double>& b, std::vector<double>& c, size_t m, int64_t& dt_01){
        const auto t0 = std::chrono::high_resolution_clock::now();
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
    const auto t1 = std::chrono::high_resolution_clock::now();
    dt_01 = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
    return 305;
}


template<>
int matrixMult_Avx(std::vector<float>& a, std::vector<float>& b, std::vector<float>& c, size_t m, int64_t& dt_01){
    const auto t0 = std::chrono::high_resolution_clock::now();
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
    const auto t1 = std::chrono::high_resolution_clock::now();
    dt_01 = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
    return 305;
}

//******************************************************************************************

template<>
    int matrixMultTransposeOpt_Avx(std::vector<double>& a, std::vector<double>& b, std::vector<double>& c, size_t m, int64_t& dt_01){
        const auto t0 = std::chrono::high_resolution_clock::now();
  //prova su double
    __m256d A,B,result,shuf,shuf2;
    for (size_t i =0; i<m; i++){
       //result = _mm256_setzero_pd();
       for(size_t j = 0; j < m; j ++){
        
          for(size_t q=0; q<m; q += 4){
            A = _mm256_loadu_pd(&a[i*m+q]);
            B = _mm256_loadu_pd(&b[i*m+q]);
            result = _mm256_mul_pd(A, B);
            shuf = _mm256_hadd_pd(result, result);
            shuf2 = _mm256_hadd_pd(shuf, shuf);
          }
          c[j+i*m]= _mm256_cvtsd_f64(shuf2);
          
        }
    }
    const auto t1 = std::chrono::high_resolution_clock::now();
    dt_01 = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
    return 306;
}


template<>
int matrixMultTransposeOpt_Avx(std::vector<float>& a, std::vector<float>& b, std::vector<float>& c, size_t m, int64_t& dt_01){
    const auto t0 = std::chrono::high_resolution_clock::now();
  //prova su double
    __m256 A,B,result,shuf,shuf2;
    __m128 shuf3;
    for (size_t i =0; i<m; i++){
       //result = _mm256_setzero_pd();
       for(size_t j = 0; j < m; j ++){
        
          for(size_t q=0; q<m; q += 8){
            A = _mm256_loadu_ps(&a[i*m+q]);
            B = _mm256_loadu_ps(&b[i*m+q]);
            result = _mm256_mul_ps(A, B);
            shuf = _mm256_hadd_ps(result, result);
            shuf2 = _mm256_hadd_ps(shuf, shuf);
            shuf3 = _mm_hadd_ps(_mm256_castps256_ps128(shuf2), _mm256_extractf128_ps(shuf2, 1));
          }
          c[j+i*m]= _mm_cvtss_f32(shuf3);
          
        }
    }
    const auto t1 = std::chrono::high_resolution_clock::now();
    dt_01 = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
    return 306;
}




