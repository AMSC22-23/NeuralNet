#include <iostream>
#include <vector>
#include <set>
#include <algorithm>
#include <random>
#include <chrono>   //per le funzioni di timing
#include <map>
#include <unordered_map>
#include <tuple>
#include <memory>
#include <cassert>
#include <utility>
#include <functional>
#include <array>
#include <iomanip>
#include <fstream>  //per output su file
#include <immintrin.h> //intrinsics intel per SIMD


template<typename T>
class MatrixSkltn{
public:
    MatrixSkltn() : n_rows(0), n_cols(0), n_nzrs(0) {};
    size_t nrows() const {return n_rows;}
    size_t ncols() const {return n_cols;}
    size_t nnzrs() const {return n_nzrs;}

    void print(std::ostream& os = std::cout) const {
    os << "nrows: " << n_rows << " | ncols:" << n_cols << " | nnz: " << n_nzrs << std::endl;
    _print(os);
  };

    virtual const T& operator()(size_t i, size_t j) const = 0;
    virtual T& operator()(size_t i, size_t j) = 0;

    virtual ~MatrixSkltn() = default;
protected:
    size_t n_rows, n_cols, n_nzrs;
    virtual void _print(std::ostream& os) const = 0;


};

template<typename T>
class Matrix : public MatrixSkltn<T> {
public:
    virtual T& operator()(size_t i, size_t j) override {
    if (m_data.size() < i + 1) {
      m_data.resize(i + 1);
      MatrixSkltn<T>::n_rows = i + 1;
    }
    const auto it = m_data[i].find(j);
    if (it == m_data[i].end()) {
      MatrixSkltn<T>::n_cols = std::max(MatrixSkltn<T>::n_cols, j + 1);
      MatrixSkltn<T>::n_nzrs++;
      return (*m_data[i].emplace(j, 0).first).second;
    }
    return (*it).second;
  }
  virtual const T& operator()(size_t i, size_t j) const override {
    return m_data[i].at(j);
  }

    virtual ~Matrix() override = default;

protected:
    virtual void _print(std::ostream &os) const{
        for (size_t i = 0; i < m_data.size(); ++i) {
                for (const auto& [j, v] : m_data[i]) {
                    os <<std::fixed << std::setprecision(2) << v << " ";
                }
                os << std::endl;
            }
    }

private:
  std::vector<std::map<size_t, T>> m_data;

};

template<typename T>
class MatrixVect : public MatrixSkltn<T> {
  public:
    virtual T& operator()(size_t i, size_t j) override {
    if (m_data.size() < i + 1) {
      m_data.resize(i + 1);
      MatrixSkltn<T>::n_rows = i + 1;
    }
    if (m_data[i].size() < j + 1) {
      m_data[i].resize(j+1);
      MatrixSkltn<T>::n_cols = j+1;
      MatrixSkltn<T>::n_nzrs++;
    }
    return m_data[i][j];
  }
  virtual const T& operator()(size_t i, size_t j) const override {
    return m_data[i].at(j);
  }

  virtual ~MatrixVect() override = default;

  protected:
    virtual void _print(std::ostream &os) const{
        for (size_t i = 0; i < m_data.size(); ++i) {
                for (size_t j = 0; j < m_data[i].size(); ++j) {
                    os <<std::fixed << std::setprecision(2) << m_data[i][j] << " ";
                }
                os << std::endl;
            }
    }
  
  private:
  std::vector<std::vector<T>> m_data; 


};

void fillMatrixDouble(Matrix<double>& x, const int m, const int n){
    using T = double;
    std::srand(static_cast<unsigned>(std::time(0)));
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            x(i,j) = static_cast<T>(std::rand()) / static_cast<T>(RAND_MAX) * 100.0;
        }
    }    

}

template<typename T>
void fillMatrix(Matrix<T>& x, size_t m, size_t n){
  std::srand(static_cast<unsigned>(std::time(0)));
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            x(i,j) = static_cast<T>(std::rand()) / static_cast<T>(RAND_MAX) * 100.0;
        }
    }   
}

template<typename T>
void fillDefaultMatrix(Matrix<T>& x, size_t m){
  x(m-1,m-2) = 1.0;
  x(m-1,m-1) = -2.0;
  x(0,0) = -2.0;
  x(0,1) = 1;
  for(size_t i = 1; i < m-1; i++){
    x(i,i) = -2.0;
    x(i,i-1) = 1;
    x(i,i+1) = 1;
  }

}

template<typename T>
void fillMatrixVect(MatrixVect<T>& x, const int m, const int n){
  std::srand(static_cast<unsigned>(std::time(0)));
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            x(i,j) = static_cast<T>(std::rand()) / static_cast<T>(RAND_MAX) * 100.0;
        }
    }   
}

template<typename T>
void fillDefaultMatrixVect(MatrixVect<T>& x, size_t m){
  x(0,m-1) = 0.0;
  x(m-1,m-2) = 1.0;
  x(m-1,m-1) = -2.0;
  x(0,0) = -2.0;
  x(0,1) = 1;
  for(size_t i = 1; i < m-1; i++){
    if(i < m-2){
      x(i, m-1) = 0.0;
    }
    x(i,i) = -2.0;
    x(i,i-1) = 1;
    x(i,i+1) = 1;
  }

}




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


//definisco funzione che riduce la matrice ad un vettore
template<typename T>
std::vector<T> vectorReduction(MatrixVect<T>& x, size_t m, size_t n){
  std::vector<T> c;
  c.resize(m*n);
  for(size_t i = 0; i < m; i++){
    for(size_t j = 0; j < n; j++){
      c[m*i+j] = x(i,j);
    }
  }
  return c;
}


//funzione SIMD per il calcolo della moltiplicazione
/**template<typename T>
void matrixMultAvx(std::vector<T>& A, std::vector<T>& B, std::vector<T>& C, size_t m, size_t n, size_t q){
  for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            __m256d result = _mm256_setzero_pd();

            for (size_t k = 0; k < q; ++k) {
                __m256d a = _mm256_loadu_pd(&A[i * q + k]);  // Carica la colonna i-esima di A
                //__m256d a = _mm256_load_pd(&A[i + k * m]);
                __m256d b = _mm256_broadcast_sd(&B[k * n + j]);  // Broadcast l'elemento j-esimo di B
              
                result = _mm256_add_pd(result, _mm256_mul_pd(a, b));  // Moltiplicazione e somma
            }

            _mm256_storeu_pd(&C[i * n + j], result);  // Salva il risultato in C
        }
    }
}**/

void matrixMultD_Avx(std::vector<double>& a, std::vector<double>& b, std::vector<double>& c, size_t m){
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




void matrixMultF_Avx(std::vector<float>& a, std::vector<float>& b, std::vector<float>& c, size_t m){
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
/**  AVX 512 NON SUPPORTATO... 
void matrixMultF_V2_Avx(std::vector<float>& a, std::vector<float>& b, std::vector<float>& c, size_t m){
  //prova su double
    __m512 A,B,result;
    for (size_t i =0; i<m; i++){
       //result = _mm256_setzero_pd();
       for(size_t q = 0; q < m; q +=16){
        result = _mm512_setzero_ps();
          for(size_t j=0; j<m; j++){
            A = _mm512_set1_ps(&a[j+i*m]);
            //A = _mm512_set1_ps(*(&a[j+i*m]));
            B = _mm512_loadu_ps(&b[j*m+q]);
            result =  _mm512_add_ps(result, _mm512_mul_ps(A, B));
          }
          _mm512_storeu_ps(&c[i*m+q], result);
        }
    }
}
**/

int main(int argc, char **argv){
  using namespace std::chrono;
    Matrix<double> mat, mat2, mat3, matR, mat2R, mat3R;
    Matrix<float> mat4, mat5, mat6, mat4R, mat5R, mat6R;
    MatrixVect<double> mat7, mat8, mat9, mat7R, mat8R, mat9R, testA, testB;
    MatrixVect<float> mat10, mat11, mat12, mat10R, mat11R, mat12R;
    std::vector<float> a,c;
    std::vector<double> d,f;
    size_t m;
    std::vector<size_t> n;
    std::ofstream outputFile("results.txt", std::ios::app);
    std::ofstream outputFileMatrix("matrix.txt");
 

    n.resize(argc);
    if (argc < 2){
      m = 5;
      //n[0]=m;
    }else{
       // for(int i = 1; i<argc; i++){
       // n[i-1] = static_cast<size_t>(std::stoi(argv[i]));
      //}
      m= static_cast<size_t>(std::stoi(argv[1]));

    }
    std::cout << "Building different test Matrix ..." << std::endl;
    std::cout << std::endl;

    fillDefaultMatrix<double>(mat, m);
    //fillDefaultMatrix<double>(mat2, m);
    fillDefaultMatrix<float>(mat4, m);
    //fillDefaultMatrix<float>(mat5, m);

    fillMatrix<double>(matR, m, m);
    //fillMatrix<double>(mat2R, m, m);
    fillMatrix<float>(mat4R, m, m);
    //fillMatrix<float>(mat5R, m, m);

    fillDefaultMatrixVect<double>(mat7, m);
    //fillDefaultMatrixVect<double>(mat8, m);
    fillDefaultMatrixVect<float>(mat10, m);
    //fillDefaultMatrixVect<float>(mat11, m);**/

    fillMatrixVect<double>(mat7R, m, m);
    //fillMatrixVect<double>(mat8R, m, m);
    fillMatrixVect<float>(mat10R, m, m);
    //fillMatrixVect<float>(mat11R, m, m);

   // for(int p =0; p<argc-1;p++){
     // m=n[p];
      if(m < 401){
        const auto t0 = high_resolution_clock::now();   //vector/map impiega troppo tempo
        mat3 = matrixProd<double>(mat, mat);
        const auto t1 = high_resolution_clock::now();
        mat6 = matrixProd<float>(mat4, mat4);
        const auto t2 = high_resolution_clock::now();
        mat3R = matrixProd<double>(matR, matR);
        const auto t3 = high_resolution_clock::now();
        mat6R = matrixProd<float>(mat4R, mat4R);
        const auto t4 = high_resolution_clock::now();
        const auto dt_01 = duration_cast<milliseconds>(t1 - t0).count();
        const auto dt_02 = duration_cast<milliseconds>(t2 - t1).count();
        const auto dt_03 = duration_cast<milliseconds>(t3 - t2).count();
        const auto dt_04 = duration_cast<milliseconds>(t4 - t3).count();
        std::cout << "time to run a " << m << " x " << m << " matrix(default Vector/Map) multiplication of double: " << dt_01 << " ms" << std::endl;
        std::cout << "time to run a " << m << " x " << m << " matrix(default Vector/Map) multiplication of float: " << dt_02 << " ms" << std::endl;
        std::cout << "************************************************" << std::endl;
        outputFile << "time to run a " << m << " x " << m << " matrix(default Vector/Map) multiplication of double: " << dt_01 << " ms" << std::endl;
        outputFile << "time to run a " << m << " x " << m << " matrix(default Vector/Map) multiplication of float: " << dt_02 << " ms" << std::endl;
        outputFile << "************************************************" << std::endl;
        std::cout << "time to run a " << m << " x " << m << " matrix(random Vector/Map) multiplication of double: " << dt_03 << " ms" << std::endl;
        std::cout << "time to run a " << m << " x " << m << " matrix(random Vector/Map) multiplication of float: " << dt_04 << " ms" << std::endl;
        std::cout << "************************************************" << std::endl;
        outputFile << "time to run a " << m << " x " << m << " matrix(random Vector/Map) multiplication of double: " << dt_03 << " ms" << std::endl;
        outputFile << "time to run a " << m << " x " << m << " matrix(random Vector/Map) multiplication of float: " << dt_04 << " ms" << std::endl;
        outputFile << "************************************************" << std::endl;
      }else{
        std::cout << "Life is too short to waste time waiting for a lezy cpu feeded by a bad memory allocation: vector-map resolution disabled..." << std::endl;
      }
      if(m<1001){
          const auto t4 = high_resolution_clock::now();
          mat9 = matrixProdVect<double>(mat7, mat7);
          const auto t5 = high_resolution_clock::now();
          mat12 = matrixProdVect<float>(mat10, mat10);
          const auto t6 = high_resolution_clock::now();
          mat9R = matrixProdVect<double>(mat7R, mat7R);
          const auto t7 = high_resolution_clock::now();
          mat12R = matrixProdVect<float>(mat10R, mat10R);
          const auto t8 = high_resolution_clock::now();

          
          
          const auto dt_05 = duration_cast<milliseconds>(t5 - t4).count();
          const auto dt_06 = duration_cast<milliseconds>(t6 - t5).count();
          const auto dt_07 = duration_cast<milliseconds>(t7 - t6).count();
          const auto dt_08 = duration_cast<milliseconds>(t8 - t7).count();

          std::cout << "time to run a " << m << " x " << m << " matrix(default Vector/Vector) multiplication of double: " << dt_05 << " ms" << std::endl;
          std::cout << "time to run a " << m << " x " << m << " matrix(default Vector/Vector) multiplication of float: " << dt_06 << " ms" << std::endl;
          std::cout << "************************************************" << std::endl;
          outputFile << "time to run a " << m << " x " << m << " matrix(default Vector/Vector) multiplication of double: " << dt_05 << " ms" << std::endl;
          outputFile << "time to run a " << m << " x " << m << " matrix(default Vector/Vector) multiplication of float: " << dt_06 << " ms" << std::endl;
          outputFile << "************************************************" << std::endl;
          std::cout << "time to run a " << m << " x " << m << " matrix(random Vector/Vector) multiplication of double: " << dt_07 << " ms" << std::endl;
          std::cout << "time to run a " << m << " x " << m << " matrix(random Vector/Vector) multiplication of float: " << dt_08 << " ms" << std::endl;
          std::cout << "************************************************" << std::endl;
          outputFile << "time to run a " << m << " x " << m << " matrix(random Vector/Vector) multiplication of double: " << dt_07 << " ms" << std::endl;
          outputFile << "time to run a " << m << " x " << m << " matrix(random Vector/Vector) multiplication of float: " << dt_08 << " ms" << std::endl;
          outputFile << "************************************************" << std::endl;

          outputFile << std::endl;
      }else{
        std::cout << "Also vector - vector optimisation turned off...do you really want to spen more then one hour to wait for results ??.." << std::endl;
      }


      if(m < 20){
        mat.print(outputFileMatrix);
        outputFileMatrix << std::endl;
      
        mat3.print(outputFileMatrix);
        outputFileMatrix << std::endl;
        mat4.print(outputFileMatrix);
        outputFileMatrix << std::endl;
      
        mat6.print(outputFileMatrix);
        outputFileMatrix << std::endl;
        mat7.print(outputFileMatrix);
        outputFileMatrix << std::endl;
        
        mat9.print(outputFileMatrix);
        outputFileMatrix << std::endl;
        mat10.print(outputFileMatrix);
        outputFileMatrix << std::endl;
        
        mat12.print(outputFileMatrix);
        outputFileMatrix << std::endl;

        matR.print(outputFileMatrix);
        outputFileMatrix << std::endl;
        
        mat3R.print(outputFileMatrix);
        outputFileMatrix << std::endl;
        mat4R.print(outputFileMatrix);
        outputFileMatrix << std::endl;
        
        mat6R.print(outputFileMatrix);
        outputFileMatrix << std::endl;
        mat7R.print(outputFileMatrix);
        outputFileMatrix << std::endl;
        
        mat9R.print(outputFileMatrix);
        outputFileMatrix << std::endl;
        mat10R.print(outputFileMatrix);
        outputFileMatrix << std::endl;
        
        mat12R.print(outputFileMatrix);
        outputFileMatrix << std::endl;
      }
      
    
      

      


      a.resize(m*m);
      d.resize(m*m);
      c.resize(m*m);
      f.resize(m*m);

      std::cout << "preparing Vector row-major structure..." << std::endl;
      d = vectorReduction<double>(mat7R, m, m);
      a = vectorReduction<float>(mat10R, m, m);

    
      std::cout<< "let's go with AVX...." << std::endl;
    
      const auto t9 = high_resolution_clock::now();
      matrixMultD_Avx(d,d,f,m);
      const auto t10 = high_resolution_clock::now();
      matrixMultF_Avx(a,a,c,m);
      //matrixMultF_V2_Avx(a,b,c,m);  NON SUPPORTATO DALLA MIA ARCHITETTURA

      const auto t11 = high_resolution_clock::now();
      const auto dt_9 = duration_cast<milliseconds>(t10 - t9).count();
      const auto dt_10 = duration_cast<milliseconds>(t11 - t10).count();

      std::cout << "time to run a " << m << " x " << m << " matrix(random Vector-AVX256) multiplication of double: " << dt_9 << " ms" << std::endl;
      std::cout << "time to run a " << m << " x " << m << " matrix(random VectorAVX256) multiplication of float: " << dt_10 << " ms" << std::endl;
      std::cout << "************************************************" << std::endl;
      outputFile << "time to run a " << m << " x " << m << " matrix(random Vector-AVX256) multiplication of double: " << dt_9 << " ms" << std::endl;
      outputFile << "time to run a " << m << " x " << m << " matrix(random VectorAVX256) multiplication of float: " << dt_10 << " ms" << std::endl;
      outputFile << "************************************************" << std::endl;

      outputFile << std::endl;
      outputFile << "---------------------------------------------------" << std::endl;
      outputFile << "---------------------------------------------------" << std::endl;
      outputFile << std::endl;
      if(m<20){
        outputFileMatrix<<"matrix mul result with AVX: "<<std::endl;
        for (size_t i =0; i<m;i++){
          for(size_t j =0; j<m;j++){
            outputFileMatrix << c[j+i*m] << " ";
          }
        }
        std::cout<<std::endl;
      }

   // }
    
    outputFile.close();
    outputFileMatrix.close();
  





    

    return 0;
}
