#include <iostream>
#include <fmt/core.h>
#include <mpi.h>
#include "fractal_mpi.h"
#include "palette.h"
#include <complex>
#include <cstdint>
#include <SFML/Graphics.hpp>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include "draw_text.h"
#include "arial_ttf2.h"

#ifdef _WIN32
#include <windows.h>
#endif

// en ves de usar fila de matrices s eusan filas de pixeles
//  arreglo de h*ancho de pixeles * alto de pixeles
//  no necesitan enviar buffer del rank 0 a los demas
//  cada proceso calcula su parte y al final se junta todo en el rank 0
int max_iterations = 10;

double x_min = -1.5;
double x_max = 1.5;
double y_min = -1.0;
double y_max = 1.0;

// si sea que la imagen uasnado las flechas derecha e izquierda se debe mandar al rank 0 x_min y x_max tamnbien y lo mismo para y con arriba y abajo
std::complex<double> c(-0.7, 0.27015);

uint32_t *pixel_buffer = nullptr;
uint32_t *texture_buffer = nullptr;
uint32_t *texture = nullptr;
int rank;
int nprocs;

int32_t running = 1;

int row_start;
int row_end;
int delta;
int padding;
void dibujar_texto_rank()
{
    auto text = fmt::format("Rank {}", rank);
    draw_text_to_texture(
        (unsigned char *)pixel_buffer,
        WIDTH,
        delta,
        text.c_str(),
        10,
        25,
        20);
}

void set_ui()
{
    fmt::println("Rank_{} setting un UI", rank);
    std::cout.flush(); // obliga a escribir en cmd
    texture_buffer = new uint32_t[WIDTH * HEIGHT];

    std::memset(texture_buffer, 0, WIDTH * HEIGHT * sizeof(uint32_t));
    // inicializar la ui

    sf::VideoMode desktop = sf::VideoMode::getDesktopMode();

    sf::RenderWindow window(sf::VideoMode({WIDTH, HEIGHT}), "Julia Set - SFML");

#ifdef _WIN32
    HWND hwnd = window.getNativeHandle();
    ShowWindow(hwnd, SW_MAXIMIZE);
#endif

    sf::Texture texture({WIDTH, HEIGHT});
    texture.update((const uint8_t *)texture_buffer);

    sf::Sprite sprite(texture);

    const sf::Font font(arial_ttf2::data2, arial_ttf2::data_len2);
    sf::Text text(font, "Julia Set", 24);
    text.setFillColor(sf::Color::White);
    text.setPosition({10, 10});
    text.setStyle(sf::Text::Bold);

    std::string options = "OPTIONS: [1] Serial 1 UP/DOWN: Change Iterations";
    sf::Text textOptions(font, options, 24);
    textOptions.setFillColor(sf::Color::White);
    textOptions.setStyle(sf::Text::Bold);
    textOptions.setPosition({10, window.getView().getSize().y - 40});

    // FPS
    int frame = 0;
    int fps = 0; // cuantos mas fps aumenta signifa que dibuja mucho mas rapido
    sf::Clock clockFrames;

    while (window.isOpen())
    {

        while (const std::optional event = window.pollEvent())
        {

            if (event->is<sf::Event::Closed>())
            {
                window.close();
                running = false;
            }
            else if (event->is<sf::Event::KeyReleased>())
            {
                auto evt = event->getIf<sf::Event::KeyReleased>();

                switch (evt->scancode)
                {
                case sf::Keyboard::Scan::Up:
                    max_iterations += 10;
                    break;
                case sf::Keyboard::Scan::Down:
                    max_iterations -= 10;
                    if (max_iterations < 10)
                        max_iterations = 10;
                    break;
                }
                fmt::print("Max iterations: {}\n", max_iterations);
                std::memset(texture_buffer, 0, WIDTH * HEIGHT * sizeof(uint32_t));
            }
        }
        // notificar
        int32_t data[2];
        data[0] = running;
        data[1] = max_iterations;
        MPI_Bcast(data, 2, MPI_INT, 0, MPI_COMM_WORLD);
        if (running == false)
        {
            break;
        }
        julia_mpi(x_min, y_min, x_max, y_max, row_start, row_end, pixel_buffer);
        dibujar_texto_rank();
        std::memcpy(texture_buffer, pixel_buffer, WIDTH * delta * sizeof(uint32_t));

        for (int i = 1; i < nprocs; i++)
        {
            // Calcular el tamaño real a recibir para este rank
            int recv_delta = delta;
            if (i == nprocs - 1) // último rank podría ser más pequeño
            {
                recv_delta = delta - padding;
            }

            MPI_Recv(
                pixel_buffer,    // destino: posición correcta en texture_buffer
                WIDTH * recv_delta * sizeof(uint32_t), // tamaño en bytes
                MPI_UNSIGNED,                              // tipo de dato
                i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                std::memcpy(
                    texture_buffer+i * delta * WIDTH,
                    pixel_buffer,
                    WIDTH * recv_delta * sizeof(uint32_t));
        }
        // Actualizar la textura con los nuevos datos calculados
        texture.update((const uint8_t *)texture_buffer);

        // contar FPS
        frame++;
        if (clockFrames.getElapsedTime().asSeconds() >= 1.0f)
        {
            fps = frame;
            frame = 0;
            clockFrames.restart();
        }

        // actualizar el titulo
        auto msg = fmt::format("Julia Set: Iteraciones: {}, FPS: {}", max_iterations, fps);
        text.setString(msg);
        // escribir imagen
        // stbi_write_png("fractal.png", WIDTH, HEIGHT, STBI_rgb_alpha, texture_buffer, WIDTH * 4);
        window.clear();
        {
            window.draw(sprite);
            window.draw(text);
            window.draw(textOptions);
        }
        window.display();
    }
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    init_freetype();
    delta = std::ceil(HEIGHT * 1.0 / nprocs);
    row_start = rank * delta;
    row_end = (rank + 1) * delta;
    padding = delta * nprocs - HEIGHT; // si es exacto da 0 y sino reajusta
    if (row_end > HEIGHT)
    {
        row_end = HEIGHT; // para que no se pase del limite
    }
    // ya esta multiplicado por 4
    pixel_buffer = new uint32_t[WIDTH * delta];
    std::memset(pixel_buffer, 0, WIDTH * delta * sizeof(uint32_t));
    fmt::print("Rank {}, nprocs {}, delta {}, row_start {}, row_end {}\n", rank, nprocs, delta, row_start, row_end);

    if (rank == 0)
    {
        fmt::println("Rank_0 starting UI");
        set_ui();
    }
    else
    {
        while (true)
        {
            int32_t data[2];
            MPI_Bcast(data, 2, MPI_INT, 0, MPI_COMM_WORLD);
            running = data[0];
            max_iterations = data[1];

            if (running == false)
            {
                break;
            }

            julia_mpi(x_min, y_min, x_max, y_max, row_start, row_end, pixel_buffer);
            dibujar_texto_rank();
            // Calcular tamaño real a enviar (el último rank podría ser más pequeño)
            int send_size = (row_end - row_start) * WIDTH * sizeof(uint32_t);

            MPI_Send(
                pixel_buffer,
                send_size,
                MPI_BYTE,
                0,
                0,
                MPI_COMM_WORLD);
        }
    }

    MPI_Finalize();
    return 0;
}