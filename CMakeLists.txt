cmake_minimum_required(VERSION 3.16)

#nombredle proyecto
project(fractal_mpi)

##set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -mavx2 -O3 -fopenmp")
find_package(fmt CONFIG REQUIRED)
find_package(Stb REQUIRED)
find_package(Freetype REQUIRED)

##find_package(SFML COMPONENTS Network Graphics Window Audio System CONFIG REQUIRED)
## --
set(MPI_ROOT "C:/oneAPI/mpi/latest")
set(MPI_INCLUDE "${MPI_ROOT}/include")
set(MPI_LIB "${MPI_ROOT}/lib")

include_directories(
    ${MPI_INCLUDE}
)

#generar ejectuable , por que medio se ejecutara main.cpp
add_executable(fractal_mpi main.cpp
    fractal_mpi.cpp
    palette.cpp
    draw_text.cpp
)


target_link_libraries(fractal_mpi PRIVATE
    fmt::fmt
    Freetype::Freetype
    ${MPI_LIB}/impi.lib
)

