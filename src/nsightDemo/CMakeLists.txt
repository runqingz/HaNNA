cmake_minimum_required(VERSION 3.16)
project(demo)

#enable_testing()
# Set up language settings
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED YES)
set(CMAKE_CXX_EXTENSIONS NO)

find_package(Halide REQUIRED)

add_executable(blur blur.cpp)
target_link_libraries(blur PRIVATE Halide::Halide Halide::ImageIO Halide::Tools)
