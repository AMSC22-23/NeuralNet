#include <iostream>
#include "mmmHeader.hpp"

// Function to generate a random value between 0 and 1
double generateRandomValue() {
    return static_cast<double>(std::rand()) / RAND_MAX;
}

// Function to create a matrix of size (rows x columns) with random values
std::vector<std::vector<double> > matrixMaker(std::size_t const & rows, std::size_t const & columns) {
    // Create a matrix of size rows x columns
    std::vector<std::vector<double> > matrix(rows, std::vector<double>(columns));

    // Populate the matrix with random values
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < columns; j++) {
            matrix[i][j] = generateRandomValue();
        }
    }

    return matrix;
}

// Function to print a matrix
void matrixPrinter(const std::vector<std::vector<double> >& matrix) {
    // Iterate over rows
    for (size_t i = 0; i < matrix.size(); i++) {
        // Iterate over columns
        for (size_t j = 0; j < matrix[i].size(); j++) {
            // Print matrix element with a space
            std::cout << matrix[i][j] << ' ';
        }
        // Move to the next line after each row
        std::cout << std::endl;
    }
}

// Function to multiply two matrices
std::vector<std::vector<double> > matrixMultiplier(const std::vector<std::vector<double> >& m1, const std::vector<std::vector<double> >& m2) {
    // Get dimensions of the matrices
    std::size_t rows1 = m1.size();
    std::size_t columns1 = m1[0].size();
    std::size_t rows2 = m2.size();
    std::size_t columns2 = m2[0].size();

    // Check if matrices can be multiplied
    if (columns1 != rows2) {
        std::cerr << "Error: Number of columns in the first matrix must be equal to the number of rows in the second matrix for multiplication." << std::endl;
        // Return an empty matrix in case of an error
        return std::vector<std::vector<double> >();
    }

    // Initialize the result matrix with zeros
    std::vector<std::vector<double> > result(rows1, std::vector<double>(columns2, 0.0));

    // Perform matrix multiplication
    for (size_t i = 0; i < rows1; i++) {
        for (size_t j = 0; j < columns2; j++) {
            for (size_t k = 0; k < columns1; k++) {
                // Multiply corresponding elements and accumulate the result
                result[i][j] += m1[i][k] * m2[k][j];
            }
        }
    }

    return result;
}

// Main function
int main(){

    // Seed random number generator
    std::srand(static_cast<unsigned int>(std::time(nullptr)));

    // Get user input for the size of the matrices
    std::size_t rows1, columns1, rows2, columns2;

    std::cout << "Enter the number of rows for Matrix 1: ";
    std::cin >> rows1;
    std::cout << "Enter the number of columns for Matrix 1: ";
    std::cin >> columns1;

    std::cout << "Enter the number of rows for Matrix 2 (must be equal to the number of columns in Matrix 1): ";
    std::cin >> rows2;

    // Check if the matrices can be multiplied
    while (rows2 != columns1) {
        std::cout << "Error: Number of rows in Matrix 2 must be equal to the number of columns in Matrix 1. Please enter a valid number of rows for Matrix 2: ";
        std::cin >> rows2;
    }

    std::cout << "Enter the number of columns for Matrix 2: ";
    std::cin >> columns2;
    std::cout << std::endl;


    // Create matrices
    auto m1 = matrixMaker(rows1, columns1);
    auto m2 = matrixMaker(rows2, columns2);

    // Print m1
    std::cout << "Matrix 1:" << std::endl;
    matrixPrinter(m1);
    std::cout << std::endl;

    // Print m2
    std::cout << "Matrix 2:" << std::endl;
    matrixPrinter(m2);
    std::cout << std::endl;

    // Multiply matrices
    auto result = matrixMultiplier(m1, m2);

    // Print result
    std::cout << "Result Matrix:" << std::endl;
    matrixPrinter(result);
    std::cout << std::endl;
    
    return 0;
}