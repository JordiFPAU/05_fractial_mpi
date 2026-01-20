#include <iostream>
#include <mpi.h>
#include <fmt/core.h>
#include <complex>
#include <SFML/Graphics.hpp>
#include "arial.ttf.h"
#include "fractal_mpi.h"
#include "draw_text.h"
#ifdef _WIN32
#include <windows.h>
#endif

double x_min = -1.5;
double x_max = 1.5;
double y_min = -1.0;
double y_max = 1.0;
int thread_count;

int max_iterations = 10;
std::complex<double> c(-0.7, 0.27015);
uint32_t *pixel_buffer = nullptr;
uint32_t *texture_buffer = nullptr;

int nprocs;
int rank;
int32_t running = 1;
int delta;
int row_start;
int row_end;
int padding;

int setup_ui()
{
    fmt::println("RANK_{} settign up_ui", rank);

    texture_buffer = new uint32_t[WIDTH * delta];
    std::memset(texture_buffer, 0, WIDTH * delta * sizeof(uint32_t));

    // inicializar la ui
    sf::VideoMode desktop = sf::VideoMode::getDesktopMode();
    // idea crear esa ventana y maximizarla
    sf::RenderWindow window(sf::VideoMode({WIDTH, HEIGHT}), "Julia Set - SFML");
#ifdef _WIN32
    HWND hwnd = window.getNativeHandle();
    ShowWindow(hwnd, SW_MAXIMIZE);
#endif
    sf::Texture texture({WIDTH, HEIGHT});
    texture.update((const uint8_t *)texture_buffer);

    sf::Sprite sprite(texture);
    // - escalar el sprite para llenar la ventana

    // -- textos
    const sf::Font font(arial_ttf, sizeof(arial_ttf));
    sf::Text text(font, "Julia Set", 24);
    text.setFillColor(sf::Color::White);
    text.setPosition({10, 10});
    text.setStyle(sf::Text::Bold);

    std::string options = "OPTIONS: [1] Serial 1 [2] Serial 2 [3] SIMD [4] OpenMP Regiones [5] OpenMP For [6] OpenMP For SIMD | UP/DOWN: Cambiar Iteraciones";
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
                    if (max_iterations > 10)
                        max_iterations -= 10;
                    break;
                default:
                    break;
                }
                std::memset(texture_buffer, 0, WIDTH * delta * sizeof(uint32_t));
            }
        }
        int32_t data[2];
        data[0] = running;
        data[1] = max_iterations;
        MPI_Bcast(data, 2, MPI_INT32_T, 0, MPI_COMM_WORLD);

        if (running == false)
        {
            break;
        }
        julia_mpi(x_min, y_min, x_max, y_max, row_start, row_end, pixel_buffer);
        std::memcpy(texture_buffer, pixel_buffer, WIDTH * delta * sizeof(uint32_t));

        for (int i = 1; i < nprocs; i++)
        {
            int new_delta = delta;
            if (rank == nprocs - 1)
            {
                new_delta = HEIGHT - padding;
            }

            MPI_Recv(
                texture_buffer + i * WIDTH * delta, // DESFASE -> en donde recibo
                WIDTH * new_delta, MPI_UNSIGNED,    // RECIBE
                i, 0, MPI_COMM_WORLD,               // Quien envia
                MPI_STATUS_IGNORE);
            std::memcpy(texture_buffer + i * WIDTH * delta,
                        pixel_buffer,
                        WIDTH * new_delta * sizeof(uint32_t));
        }
        texture.update((const uint8_t *)texture_buffer);

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
        window.clear();
        {
            window.draw(sprite);
            window.draw(text);
            window.draw(textOptions);
        }
        window.display();
    }
    return 0;
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    nprocs;

    rank;

    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    init_freetype(); // Inicializa la libreria freetype

    delta = std::ceil(HEIGHT * 1.0 / nprocs);
    row_start = rank * delta;
    row_end = (rank + 1) * delta;
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
        setup_ui();
    }
    else
    {
        while (true)
        {
            int32_t data[2];

            MPI_Bcast(data, 2, MPI_INT32_T, 0, MPI_COMM_WORLD);
            running = data[0];
            max_iterations = data[1];
            if (running == false)
            {
                break;
            }
            julia_mpi(x_min, y_min, x_max, y_max, row_start, row_end, pixel_buffer);
            auto text = fmt::format("Rank {}", rank);
            draw_text_to_texture((unsigned char *)pixel_buffer, WIDTH, delta, text.c_str(), 10, 25, 20);
            MPI_Send(
                pixel_buffer,
                WIDTH * delta, MPI_UNSIGNED, // envio
                0, 0, MPI_COMM_WORLD         // recibo
            );
        }
    }

    MPI_Finalize();
    return 0;
}