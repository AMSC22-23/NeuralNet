#ifndef MMM_HEADER_HPP
#define MMM_HEADER_HPP

#include <vector>

// Function declarations
std::vector<std::vector<double> > matrixMaker(std::size_t const & rows, std::size_t const & columns);
void matrixPrinter(const std::vector<std::vector<double> >& matrix);
double generateRandomValue();
std::vector<std::vector<double> > matrixMultiplier(const std::vector<std::vector<double> >& matrix1, const std::vector<std::vector<double> >& matrix2);

#endif
