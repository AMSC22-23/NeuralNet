#include "MatrixSkltn.hpp"

#ifndef MATRIX_VM_VV_H
#define MATRIX_VM_VV_H


#include <map>   //per gestire le funzioni map
#include <iomanip>   //per manipolare i numeri dopo la virgola da plotare



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

    MatrixVect(std::size_t rows, std::size_t cols, std::size_t nnz) : MatrixSkltn<T>(rows, cols, nnz){};

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
  //@note: why do you need this matrix if you have the MatrixFlat which is better this
  std::vector<std::vector<T>> m_data; 


};


#endif