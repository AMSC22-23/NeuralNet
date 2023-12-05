#include "MatrixSkltn2.hpp"
#include <iomanip>


#ifndef MATRIXVECT2_HPP
#define MATRIXVECT2_HPP

template <typename T>
class MatrixVect : public MatrixSkltn<T> {
public:
    
    
    //Costruisce una matrice di zeri
    MatrixVect(std::size_t rows, std::size_t cols) : MatrixSkltn<T>(rows, cols), m_data(rows, std::vector<T>(cols)) {}

    
    //Costruttore che riempie la matrice con valori di un vettore 
    MatrixVect(std::size_t rows, std::size_t cols, const std::vector<T>& initialData)
        : MatrixSkltn<T>(rows, cols), m_data(rows, std::vector<T>(cols)) {
        if (initialData.size() == rows * cols) {
            std::size_t index = 0;
            for (std::size_t i = 0; i < rows; ++i) {
                for (std::size_t j = 0; j < cols; ++j) {
                    m_data[i][j] = initialData[index++];
                }
            }
        } else {
            throw std::invalid_argument("MatrixVect: Incorrect size of initialData vector");
        }
    }


    //costruttore con valori random tra minValue e maxValue
    MatrixVect(std::size_t rows, std::size_t cols, T minValue, T maxValue)
        : MatrixSkltn<T>(rows, cols), m_data(rows, std::vector<T>(cols)) {
        generate_random_matrix(minValue, maxValue);
    }

    

    //stampa matrici di tipo MatrixVect
    void print(std::ostream& os = std::cout) const override {
        for (std::size_t i = 0; i < MatrixSkltn<T>::n_rows; ++i) {
            for (std::size_t j = 0; j < MatrixSkltn<T>::n_cols; ++j) {
                os << std::fixed << std::setprecision(2) << m_data[i][j] << " ";
            }
            os << std::endl;
        }
    }

    //accede ai dati read-only
    const T& operator()(size_t i, size_t j) const override {
    checkBounds(i, j);
    return m_data[i][j];
    }

    //read and write
    T& operator()(size_t i, size_t j) override {
    checkBounds(i, j);
    return m_data[i][j];
    }


private:

    //struttura dati
    std::vector<std::vector<T> > m_data;

    //riempie m_data con valori random, chiamata dal costruttore random
    void generate_random_matrix(T minValue, T maxValue) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<T> dist(minValue, maxValue);

        for (std::size_t i = 0; i < MatrixSkltn<T>::n_rows; ++i) {
            for (std::size_t j = 0; j < MatrixSkltn<T>::n_cols; ++j) {
                m_data[i][j] = dist(gen);
            }
        }
    }

    //controlla che gli indici utilizzati con gli operatori() siano corretti
    void checkBounds(std::size_t i, std::size_t j) const {
        if (i >= MatrixSkltn<T>::n_rows || j >= MatrixSkltn<T>::n_cols) {
            throw std::out_of_range("MatrixVect: Index out of bounds");
        }
    }
};

#endif
