#ifndef FRACTAL_MPI_H
#define _FRACTAL_MPI_
#include <vector>
#include <cstdint>
#define WIDTH 1600
#define HEIGHT 900

#include <complex>
extern std::complex<double> c;

void julia_mpi(double x_min, double y_min, double x_max,
                    double y_max,
                    int row_start, int row_end, uint32_t *pixel_buffer);

#endif