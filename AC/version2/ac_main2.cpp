#include <iostream>
#include <vector>
#include <set>
#include <algorithm>
#include <random>
#include <chrono>
#include <map>
#include <unordered_map>
#include <tuple>
#include <memory>
#include <cassert>
#include <utility>
#include <functional>
#include <array>
#include <iomanip>


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
                    std::cout <<std::fixed << std::setprecision(2) << v << " ";
                }
                std::cout << std::endl;
            }
    }

private:
  std::vector<std::map<size_t, T>> m_data;

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


int main(){
    Matrix<double> mat, mat2;
    for(int i = 0 ; i<5; i++){
        for(int j = 0; j<5 ; j++){
            mat(i,j) = 1.5;
        }
    }

    fillMatrixDouble(mat2, 4, 4);

    mat.print(std::cout);
    std::cout << std::endl;
    mat2.print(std::cout);

    return 0;
}
