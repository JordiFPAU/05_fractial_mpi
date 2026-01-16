#include <iostream>
#include <mpi.h>
#include <fmt/core.h>
#include <complex>

#include "fractal_mpi.h"
#include "draw_text.h"
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

double x_min = -1.5;
double x_max = 1.5;
double y_min = -1.0;
double y_max = 1.0;
int thread_count;

int max_iterations = 10;
std::complex<double> c(-0.7, 0.27015);
uint32_t *pixel_buffer = nullptr;
uint32_t *texture_buffer = nullptr;

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int nprocs;

    int rank;

    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    init_freetype(); // Inicializa la libreria freetype

    int delta = std::ceil(HEIGHT * 1.0 / nprocs);
    int row_start = rank * delta;
    int row_end = (rank + 1) * delta;
    int padding = delta * nprocs - HEIGHT;
    if (row_end > HEIGHT)
    {
        row_end = HEIGHT;
    }

    // Inicializamos los buffers
    pixel_buffer = new uint32_t[WIDTH * delta];
    std::memset(pixel_buffer, 0, WIDTH * delta * sizeof(uint32_t));
    fmt::println("Rank_{}, nprocs={}, delta = {}, start_{}, end_{}", rank, nprocs, delta, row_start, row_end);
    std::cout.flush();

    if (rank == 0)
    {
        texture_buffer = new uint32_t[WIDTH * HEIGHT];
        std::memset(texture_buffer, 0, WIDTH * HEIGHT * sizeof(uint32_t));
        julia_mpi(x_min, y_min, x_max, y_max, row_start, row_end, pixel_buffer);
        std::memcpy(texture_buffer, pixel_buffer, WIDTH * delta * sizeof(uint32_t));
        for (int i = 1; i < nprocs; i++)
        {
            MPI_Recv(
                texture_buffer + i * WIDTH * delta, // DESFASE -> en donde recibo
                WIDTH * delta, MPI_UNSIGNED,        // RECIBE
                i, 0, MPI_COMM_WORLD,               // Quien envia
                MPI_STATUS_IGNORE);
        }
        // Escribir la imagen
        stbi_write_png("fractal.png", WIDTH, HEIGHT, STBI_rgb_alpha, texture_buffer, WIDTH * 4);
    }
    else
    {
        julia_mpi(x_min, y_min, x_max, y_max, row_start, row_end, pixel_buffer);
        auto text = fmt::format("Rank {}", rank);
        draw_text_to_texture((unsigned char *)pixel_buffer, WIDTH, delta, text.c_str(), 10, 25, 20);
        MPI_Send(
            pixel_buffer,
            WIDTH * delta, MPI_UNSIGNED, // envio
            0, 0, MPI_COMM_WORLD         // recibo
        );
    }

    MPI_Finalize();
    return 0;
}