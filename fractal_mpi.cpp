#include "fractal_mpi.h"
#include <cstdint>
#include <complex>
#include "palette.h"
extern int max_iterations;

int divergente(double x, double y)
{
    int iter = 1;
    double zr = x;
    double zi = y;

    while ((zr * zr + zi * zi) < 4.0 && iter < max_iterations) // en maximo de iteraciones viene de main
    {
        double dr = zr * zr - zi * zi + c.real();
        double di = 2.0 * zr * zi + c.imag();

        zr = dr;
        zi = di;

        iter++;
    }

    if (iter < max_iterations)
    {
        int index = iter % PALETTE_SIZE;
        return color_ramp[index];
    }

    return 0xFF000000; // color negro
}
void julia_mpi(double x_min, double y_min, double x_max,
               double y_max,
               int row_start, int row_end, uint32_t *pixel_buffer)
{
    double dx = (x_max - x_min) / (WIDTH);
    double dy = (y_max - y_min) / (HEIGHT);

    for (int j = row_start; j < row_end; j++)
    {
        for (int i = 0; i < WIDTH; i++)
        {
            double x = x_min + i * dx;
            double y = y_min + j * dy;

            auto color = divergente(x, y); // auto es igual a var --> inferencia de tipos

            pixel_buffer[(j - row_start) * WIDTH + i] = color; // asignamos el color al pixel
        }
    }
    for (int i = 0; i < WIDTH; i++)
    {
        pixel_buffer[i] = 0xFF000000;
    }
}
