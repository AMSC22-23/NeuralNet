#include <iostream>
#include <vector>
#include <string>
#include "../../include/mmm.hpp"
#include "../../include/mmm_blas.hpp"
#include "../../include/MatrixFlat.hpp"
#include <chrono>


void new_multiT(const MatrixFlat<double>& A, const MatrixFlat<double>& B, MatrixFlat<double>& C, int64_t& time, int inner_tileSize) {

    std::cout<<"Performing mmm_multiT in double precision (double)"<<std::endl;


    std::size_t rows = A.nrows(), columns = B.ncols(), inners = A.ncols();
    int tileSize = rows/4;

    std::cout<<"Tile Size: "<<(tileSize)<<std::endl;
    const auto t0 = std::chrono::high_resolution_clock::now();


#pragma omp parallel for shared(A, B, C, rows, columns, inners, inner_tileSize, tileSize) default(none) \
      collapse(2) num_threads(8)


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


int main(int argc, char ** argv){

    int dim = std::stoi(argv[1]);
    int tileSize = std::stoi(argv[2]);


    MatrixFlat<double> A(dim, dim, -10, 10);
    MatrixFlat<double> B(dim, dim, -10, 10);
    MatrixFlat<double> C(dim, dim);
    MatrixFlat<double> Cblas(dim, dim);
    MatrixFlat<float> Af(dim, dim, -10, 10);
    MatrixFlat<float> Bf(dim, dim, -10, 10);
    MatrixFlat<float> Cf(dim, dim);
    MatrixFlat<float> Cblasf(dim, dim);
    MatrixFlat<float> Cf_loopI(dim, dim);

    int64_t time;

    mmm_gmultiT(A, B, C, time, tileSize);
    std::cout<<"This operation took: "<<time<< " [ms]"<<std::endl;
    mmm_blas(A, B, Cblas, time);
    std::cout<<"The same operation using openBlas took: "<<time<< " [ms]"<<std::endl;
    std::cout<<"We check if the result is the same: "<<std::endl;
    std::cout<<"nnz(C-Cblas): "<<(Cblas-C).nnzrs()<<std::endl;

    std::cout<<"-----------------------------------------------------------------------"<<std::endl;

    mmm_gmultiT(Af, Bf, Cf, time, tileSize);
    std::cout<<"This operation took: "<<time<< " [ms]"<<std::endl;
    mmm_blas(Af, Bf, Cblasf, time);
    std::cout<<"The same operation using openBlas took: "<<time<< " [ms]"<<std::endl;
    std::cout<<"We check if the result is the same: "<<std::endl;

    mmm_loopI(Af, Bf, Cf_loopI, time);

    std::cout<<"nnz(Cf-Cf_loopI): "<<(Cf-Cf_loopI).nnzrs()<<std::endl;
    std::cout<<"nnz(Cf-Cblasf): "<<(Cf-Cblasf).nnzrs()<<std::endl;

    std::cout<<"-----------------------------------------------------------------------"<<std::endl;


    return 0;

}