cmake_minimum_required(VERSION 3.16)
project(depth_separable_conv2d)

set(HEADER_FILES
    kernel.hpp)
set(SOURCE_FILES
    kernel.cpp)
#enable_testing()


# Set up language settings
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED YES)
set(CMAKE_CXX_EXTENSIONS NO)

# Find Halide
find_package(Halide REQUIRED)

add_executable(depth_separable_conv2d depth_separable_conv2d.cpp)
target_link_libraries(depth_separable_conv2d PRIVATE Halide::Halide Halide::ImageIO)
