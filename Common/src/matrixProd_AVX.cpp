#include "../include/matrixProd_AVX.hpp"
#include<chrono>
#include<iostream>



/**
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
} **/

//******************************************************************************************

/**
template<>
int matrixMultTransposeOpt_Avx(std::vector<double>& a, std::vector<double>& b_transpose, std::vector<double>& c, size_t m, int64_t& dt_01){
        const auto t0 = std::chrono::high_resolution_clock::now();
  //prova su double
    __m256d A,B,result,shuf,shuf2,partial;
    //std::vector<double> d;
    //d.resize(4);
    for (size_t i =0; i<m; i++){
       result = _mm256_setzero_pd();
       for(size_t j = 0; j < m; j ++){
            partial = _mm256_setzero_pd();
          for(size_t q=0; q<m; q += 4){
            A = _mm256_loadu_pd(&a[i*m+q]);
            B = _mm256_loadu_pd(&b_transpose[j*m+q]);
            result =  _mm256_mul_pd(A, B);
            shuf = _mm256_hadd_pd(result, result);
            
            partial = _mm256_add_pd(partial, shuf);
          }
          c[j+i*m] = ((double*)&partial)[0] + ((double*)&partial)[2];
          
        }
    }
    const auto t1 = std::chrono::high_resolution_clock::now();
    dt_01 = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
    return 306;
}



template<>
int matrixMultTransposeOpt_Avx(std::vector<float>& a, std::vector<float>& b_transpose, std::vector<float>& c, size_t m, int64_t& dt_01){
    const auto t0 = std::chrono::high_resolution_clock::now();
  //prova su double
     __m256 A,B,result,shuf,shuf2,partial;
    for (size_t i =0; i<m; i++){
       //result = _mm256_setzero_pd();
       for(size_t j = 0; j < m; j ++){
            partial = _mm256_setzero_ps();
          for(size_t q=0; q<m; q += 8){
            A = _mm256_loadu_ps(&a[i*m+q]);
            B = _mm256_loadu_ps(&b_transpose[j*m+q]);
            result =  _mm256_mul_ps(A, B);
            shuf = _mm256_hadd_ps(result, result);
            //shuf2 = _mm256_hadd_pd(shuf, shuf);
            
            partial = _mm256_add_ps(partial, shuf);
             
          }
          
             
          c[j+i*m] = ((float*)&partial)[0] + ((float*)&partial)[1]+((float*)&partial)[4] + ((float*)&partial)[5];
          
        }
    }
    const auto t1 = std::chrono::high_resolution_clock::now();
    dt_01 = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
    return 306;
}
**/

template<>
    int matrixMult_Avx(std::vector<double>& a, std::vector<double>& b, std::vector<double>& c,size_t ma, size_t na, size_t nb, int64_t& dt_01){
        const auto t0 = std::chrono::high_resolution_clock::now();
  //prova su double
    __m256d A,B,result;
    for (size_t i =0; i<ma; i++){
       //result = _mm256_setzero_pd();
       for(size_t q = 0; q <= na; q +=4){
        result = _mm256_setzero_pd();
          for(size_t j=0; j<na; j++){
            A = _mm256_broadcast_sd(&a[j+i*na]);
            B = _mm256_loadu_pd(&b[j*nb+q]);
            result =  _mm256_add_pd(result, _mm256_mul_pd(A, B));
          }
          _mm256_storeu_pd(&c[i*nb+q], result);
        }
    }
    const auto t1 = std::chrono::high_resolution_clock::now();
    dt_01 = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
    return 305;
}


template<>
int matrixMult_Avx(std::vector<float>& a, std::vector<float>& b, std::vector<float>& c, size_t ma, size_t na, size_t nb, int64_t& dt_01){
    const auto t0 = std::chrono::high_resolution_clock::now();
  //prova su double
    __m256 A,B,result;
    for (size_t i =0; i<ma; i++){
       //result = _mm256_setzero_pd();
       for(size_t q = 0; q <= na; q +=8){
        result = _mm256_setzero_ps();
          for(size_t j=0; j<na; j++){
            A = _mm256_broadcast_ss(&a[j+i*na]);
            B = _mm256_loadu_ps(&b[j*nb+q]);
            //std::cout << "index A: " << j+i*na << " val A: " << ((float*)&A)[0] << " " << ((float*)&A)[1] << " " << ((float*)&A)[2] << " " << ((float*)&A)[3]<< " " << ((float*)&A)[4] << " " << ((float*)&A)[5] << " " << ((float*)&A)[6]<< " " << ((float*)&A)[7]  << " index B: " <<j*nb+q << " val B: " << ((float*)&B)[0] << " " << ((float*)&B)[1]<< " " << ((float*)&B)[2] << " " << ((float*)&B)[3]<< " " << ((float*)&B)[4]<< " " << ((float*)&B)[5] << " " << ((float*)&B)[6]<< " " << ((float*)&B)[7] << std::endl;
            result =  _mm256_add_ps(result, _mm256_mul_ps(A, B));
          }
          _mm256_storeu_ps(&c[i*nb+q], result);
        }
    }
    const auto t1 = std::chrono::high_resolution_clock::now();
    dt_01 = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
    return 305;
}



template<>
int matrixMultTransposeOpt_Avx(std::vector<double>& a, std::vector<double>& b_transpose, std::vector<double>& c, size_t ma, size_t na, size_t nb,  int64_t& dt_01){
        const auto t0 = std::chrono::high_resolution_clock::now();
  //prova su double
    __m256d A,B,result,shuf,shuf2,partial;
    //std::vector<double> d;
    //d.resize(4);
    for (size_t i =0; i<ma; i++){
       result = _mm256_setzero_pd();
       for(size_t j = 0; j < nb; j ++){
            partial = _mm256_setzero_pd();
          for(size_t q=0; q<na; q += 4){
            A = _mm256_loadu_pd(&a[i*na+q]);
            B = _mm256_loadu_pd(&b_transpose[j*na+q]);
            result =  _mm256_mul_pd(A, B);
            shuf = _mm256_hadd_pd(result, result);
            
            partial = _mm256_add_pd(partial, shuf);
          }
          c[j+i*nb] = ((double*)&partial)[0] + ((double*)&partial)[2];
          
        }
    }
    const auto t1 = std::chrono::high_resolution_clock::now();
    dt_01 = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
    return 306;
}

template<>
int matrixMultTransposeOpt_Avx(std::vector<float>& a, std::vector<float>& b_transpose, std::vector<float>& c, size_t ma, size_t na, size_t nb, int64_t& dt_01){
    const auto t0 = std::chrono::high_resolution_clock::now();
  //prova su double
     __m256 A,B,result,shuf,shuf2,partial;
    for (size_t i =0; i<ma; i++){
       //result = _mm256_setzero_pd();
       for(size_t j = 0; j < nb; j ++){
            partial = _mm256_setzero_ps();
          for(size_t q=0; q<na; q += 8){
            A = _mm256_loadu_ps(&a[i*na+q]);
            B = _mm256_loadu_ps(&b_transpose[j*na+q]);
            //std::cout << "index A: " << i*(na+da)+q << " val A: " << ((float*)&A)[0] << " " << ((float*)&A)[1] << " " << ((float*)&A)[2] << " " << ((float*)&A)[3]<< " " << ((float*)&A)[4] << " " << ((float*)&A)[5] << " " << ((float*)&A)[6]<< " " << ((float*)&A)[7]  << " index B: " <<j*(na+da)+q << " val B: " << ((float*)&B)[0] << " " << ((float*)&B)[1]<< " " << ((float*)&B)[2] << " " << ((float*)&B)[3]<< " " << ((float*)&B)[4]<< " " << ((float*)&B)[5] << " " << ((float*)&B)[6]<< " " << ((float*)&B)[7] << std::endl;
            result =  _mm256_mul_ps(A, B);
            shuf = _mm256_hadd_ps(result, result);
            //shuf2 = _mm256_hadd_pd(shuf, shuf);
            
            partial = _mm256_add_ps(partial, shuf);
             
          }
          
             
          c[j+i*nb] = ((float*)&partial)[0] + ((float*)&partial)[1]+((float*)&partial)[4] + ((float*)&partial)[5];
          
        }
    }
    const auto t1 = std::chrono::high_resolution_clock::now();
    dt_01 = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
    return 306;
}




