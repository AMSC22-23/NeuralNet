#include "../include/mmm.hpp"
#include <cblas.h>
#include <chrono>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <sstream>



void mmm_naive(const MatrixFlat<double>& A, const MatrixFlat<double>& B, MatrixFlat<double>& C, int64_t& time){

    std::cout<<"Performing naive mmm in double precision (double)  "<<std::endl;
    std::size_t rows = A.nrows(), columns = B.ncols(), inners = A.ncols();


    const auto t0 = std::chrono::high_resolution_clock::now();

    for (std::size_t row = 0; row < rows; row++) {
        for (std::size_t col = 0; col < columns; col++) {
            for (std::size_t inner = 0; inner < inners; inner++) {
                C[row * columns + col] +=
                        A[row * columns + inner] * B[inner * columns + col];
            }
        }
    }

    const auto t1 = std::chrono::high_resolution_clock::now();
    time = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();


};

void mmm_naive(const MatrixFlat<float>& A, const MatrixFlat<float>& B, MatrixFlat<float>& C, int64_t& time){

    std::cout<<"Performing naive mmm in single precision (float) "<<std::endl;
    std::size_t rows = A.nrows(), columns = B.ncols(), inners = A.ncols();


    const auto t0 = std::chrono::high_resolution_clock::now();

    for (std::size_t row = 0; row < rows; row++) {
        for (std::size_t col = 0; col < columns; col++) {
            for (std::size_t inner = 0; inner < inners; inner++) {
                C[row * columns + col] +=
                        A[row * columns + inner] * B[inner * columns + col];
            }
        }
    }

    const auto t1 = std::chrono::high_resolution_clock::now();
    time = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();


};

void mmm_naive_RegisterAcc(const MatrixFlat<double>& A, const MatrixFlat<double>& B, MatrixFlat<double>& C, int64_t& time){

    std::cout<<"Performing naive_mmm_RegisterAcc in double precision (double) "<<std::endl;

    std::size_t rows = A.nrows(), columns = B.ncols(), inners = A.ncols();


    const auto t0 = std::chrono::high_resolution_clock::now();

    for (std::size_t row = 0; row < rows; row++) {
        for (std::size_t col = 0; col < columns; col++) {
            double acc = 0;
            for (std::size_t inner = 0; inner < inners; inner++) {
                acc += A[row * columns + inner] * B[inner * columns + col];
            }
            C[row * columns + col] =  acc;
        }
    }

    const auto t1 = std::chrono::high_resolution_clock::now();
    time = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();



};

void mmm_naive_RegisterAcc(const MatrixFlat<float>& A, const MatrixFlat<float>& B, MatrixFlat<float>& C, int64_t& time){


    std::cout<<"Performing naive_mmm_RegisterAcc in single precision (float) "<<std::endl;

    std::size_t rows = A.nrows(), columns = B.ncols(), inners = A.ncols();

    const auto t0 = std::chrono::high_resolution_clock::now();

    for (std::size_t row = 0; row < rows; row++) {
        for (std::size_t col = 0; col < columns; col++) {
            float acc = 0;
            for (std::size_t inner = 0; inner < inners; inner++) {
                acc += A[row * columns + inner] * B[inner * columns + col];
            }
            C[row * columns + col] =  acc;
        }
    }

    const auto t1 = std::chrono::high_resolution_clock::now();
    time = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();



};

void mmm_loopI(const MatrixFlat<double>& A, const MatrixFlat<double>& B, MatrixFlat<double>& C, int64_t& time) {

    std::cout<<"Performing mmm_loopI in double precision (double) "<<std::endl;

    std::size_t rows = A.nrows(), columns = B.ncols(), inners = A.ncols();

    const auto t0 = std::chrono::high_resolution_clock::now();

    for (std::size_t row = 0; row < rows; row++) {
        for (std::size_t inner = 0; inner < inners; inner++){
            for (std::size_t col = 0; col < columns; col++)
             {
                C[row * columns + col] += A[row * columns + inner] * B[inner * columns + col];
            }
        }
    }

    const auto t1 = std::chrono::high_resolution_clock::now();
    time = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();



}

void mmm_loopI(const MatrixFlat<float>& A, const MatrixFlat<float>& B, MatrixFlat<float>& C, int64_t& time){

    std::cout<<"Performing mmm_loopI in single precision (float) "<<std::endl;

    std::size_t rows = A.nrows(), columns = B.ncols(), inners = A.ncols();

    const auto t0 = std::chrono::high_resolution_clock::now();

    for (std::size_t row = 0; row < rows; row++) {
        for (std::size_t inner = 0; inner < inners; inner++){
            for (std::size_t col = 0; col < columns; col++)
            {
                C[row * columns + col] += A[row * columns + inner] * B[inner * columns + col];
            }
        }
    }

    const auto t1 = std::chrono::high_resolution_clock::now();
    time = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();


};

void mmm_tiling(const MatrixFlat<double>& A, const MatrixFlat<double>& B, MatrixFlat<double>& C, int64_t& time, int tileSize){

    std::cout<<"Performing mmm_tiling in double precision (double)"<<std::endl;

    std::size_t rows = A.nrows(), columns = B.ncols(), inners = A.ncols();

    const auto t0 = std::chrono::high_resolution_clock::now();

        for (int innerTile = 0; innerTile < inners; innerTile += tileSize) {
            for (int row = 0; row < rows; row++) {
                int innerTileEnd = std::min<int>(inners, innerTile + tileSize);
                for (int inner = innerTile; inner < innerTileEnd; inner++) {
                    for (int column = 0; column < columns; column++) {
                        C[row * columns + column] +=
                                A[row * inners + inner] * B[inner * columns + column];
                    }
                }
            }
        }

    const auto t1 = std::chrono::high_resolution_clock::now();
    time = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();

};

void mmm_tiling(const MatrixFlat<float>& A, const MatrixFlat<float>& B, MatrixFlat<float>& C, int64_t& time, int tileSize){

    std::cout<<"Performing mmm_tiling in single precision (single)"<<std::endl;

    std::size_t rows = A.nrows(), columns = B.ncols(), inners = A.ncols();

    const auto t0 = std::chrono::high_resolution_clock::now();

    for (int innerTile = 0; innerTile < inners; innerTile += tileSize) {
        for (int row = 0; row < rows; row++) {
            int innerTileEnd = std::min<int>(inners, innerTile + tileSize);
            for (int inner = innerTile; inner < innerTileEnd; inner++) {
                for (int column = 0; column < columns; column++) {
                    C[row * columns + column] +=
                            A[row * inners + inner] * B[inner * columns + column];
                }
            }
        }
    }

    const auto t1 = std::chrono::high_resolution_clock::now();
    time = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();

};

void mmm_multiT(const MatrixFlat<float>& A, const MatrixFlat<float>& B, MatrixFlat<float>& C, int64_t& time, int tileSize){

    std::cout<<"Performing mmm_multiT in single precision (single)"<<std::endl;

    std::size_t rows = A.nrows(), columns = B.ncols(), inners = A.ncols();

    const auto t0 = std::chrono::high_resolution_clock::now();

//@note: do not hard code the number of threads
#pragma omp parallel for shared(A, B, C, rows, columns, inners, tileSize) default(none) \
  collapse(2) num_threads(8)


    for (int rowTile = 0; rowTile < rows; rowTile += 256) {
        for (int columnTile = 0; columnTile < columns; columnTile += 256) {
            for (int innerTile = 0; innerTile < inners; innerTile += tileSize) {
                for (int row = rowTile; row < rowTile + 256; row++) {
                    int innerTileEnd = std::min<int>(inners, innerTile + tileSize);
                    for (int inner = innerTile; inner < innerTileEnd; inner++) {
                        for (int col = columnTile; col < columnTile + 256; col++) {
                            C[row * columns + col] +=
                                    A[row * inners + inner] * B[inner * columns + col];
                        } } } } } }
    const auto t1 = std::chrono::high_resolution_clock::now();
    time = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
};

void mmm_multiT(const MatrixFlat<double>& A, const MatrixFlat<double>& B, MatrixFlat<double>& C, int64_t& time, int tileSize) {

    std::cout<<"Performing mmm_multiT in double precision (double)"<<std::endl;


    std::size_t rows = A.nrows(), columns = B.ncols(), inners = A.ncols();

    const auto t0 = std::chrono::high_resolution_clock::now();

    #pragma omp parallel for shared(A, B, C, rows, columns, inners, tileSize) default(none) \
      collapse(2) num_threads(8)



    for (int rowTile = 0; rowTile < rows; rowTile += 256) {
        for (int columnTile = 0; columnTile < columns; columnTile += 256) {
            for (int innerTile = 0; innerTile < inners; innerTile += tileSize) {
                for (int row = rowTile; row < rowTile + 256; row++) {
                    int innerTileEnd = std::min<int>(inners, innerTile + tileSize);
                    for (int inner = innerTile; inner < innerTileEnd; inner++) {
                        for (int col = columnTile; col < columnTile + 256; col++) {
                            C[row * columns + col] +=
                                    A[row * inners + inner] * B[inner * columns + col];
                        } } } } } }

    const auto t1 = std::chrono::high_resolution_clock::now();
    time = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
}


void mmm_gmultiT(const MatrixFlat<double>& A, const MatrixFlat<double>& B, MatrixFlat<double>& C, int64_t& time, int inner_tileSize,  int num_threads) {

    std::cout<<"Performing mmm_gmultiT in double precision (double)"<<std::endl;


    std::size_t rows = A.nrows(), columns = B.ncols(), inners = A.ncols();
    int tileSize = rows/4;

    std::cout<<"Tile Size: "<<(tileSize)<<std::endl;
    const auto t0 = std::chrono::high_resolution_clock::now();


#pragma omp parallel for shared(A, B, C, rows, columns, inners, inner_tileSize, tileSize) default(none) \
      collapse(2) num_threads(num_threads)


    for (int rowTile = 0; rowTile < rows; rowTile += tileSize) {
        for (int columnTile = 0; columnTile < columns; columnTile += tileSize) {
            for (int innerTile = 0; innerTile < inners; innerTile += inner_tileSize) {
                for (int row = rowTile; row < std::min<int>(rowTile + tileSize , rows); row++) {
                    int innerTileEnd = std::min<int>(inners, innerTile + inner_tileSize);
                    for (int inner = innerTile; inner < innerTileEnd; inner++) {
                        for (int col = columnTile; col < std::min<int>(columnTile + tileSize, columns); col++) {
                            C[row * columns + col] +=
                                    A[row * inners + inner] * B[inner * columns + col];
                        } } } } } }


    const auto t1 = std::chrono::high_resolution_clock::now();
    time = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();


}


void mmm_gmultiT(const MatrixFlat<float>& A, const MatrixFlat<float>& B, MatrixFlat<float>& C, int64_t& time, int inner_tileSize, int num_threads) {

    std::cout<<"Performing mmm_gmultiT in single precision (float)"<<std::endl;


    std::size_t rows = A.nrows(), columns = B.ncols(), inners = A.ncols();
    int tileSize = rows/4;

    std::cout<<"Tile Size: "<<(tileSize)<<std::endl;
    const auto t0 = std::chrono::high_resolution_clock::now();


#pragma omp parallel for shared(A, B, C, rows, columns, inners, inner_tileSize, tileSize) default(none) \
      collapse(2) num_threads(num_threads)


    for (int rowTile = 0; rowTile < rows; rowTile += tileSize) {
        for (int columnTile = 0; columnTile < columns; columnTile += tileSize) {
            for (int innerTile = 0; innerTile < inners; innerTile += inner_tileSize) {
                for (int row = rowTile; row < std::min<int>(rowTile + tileSize , rows); row++) {
                    int innerTileEnd = std::min<int>(inners, innerTile + inner_tileSize);
                    for (int inner = innerTile; inner < innerTileEnd; inner++) {
                        for (int col = columnTile; col < std::min<int>(columnTile + tileSize, columns); col++) {
                            C[row * columns + col] +=
                                    A[row * inners + inner] * B[inner * columns + col];
                        } } } } } }


    const auto t1 = std::chrono::high_resolution_clock::now();
    time = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();


}



void appendCSVRow(const std::vector<std::string>& rowData,  bool newline ){
    std::ofstream file;
    std::string filename = "profiling_results.csv";
    file.open(filename, std::ios_base::app); // Apre il file in modalità append

    if (!file.is_open()) {
        std::cerr << "Cannot open file: " << filename << std::endl;
        return;
    }

    // Costruisce una stringa con i dati da aggiungere
    std::ostringstream oss;
    for (size_t i = 0; i < rowData.size(); ++i) {
        oss << rowData[i];
        if (i != rowData.size() - 1) {
            oss << ","; // Aggiunge la virgola tra i valori
        }
    }
    if(newline)
     oss << "\n"; // Aggiunge una nuova riga alla fine

    file << oss.str(); // Aggiunge la riga al file

    file.close(); // Chiude il file dopo aver aggiunto la riga
}







