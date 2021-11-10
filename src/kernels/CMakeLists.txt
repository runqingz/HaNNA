cmake_minimum_required(VERSION 3.16)
project(kernels)

#enable_testing()
# Set up language settings
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED YES)
set(CMAKE_CXX_EXTENSIONS NO)

find_package(Halide REQUIRED)

add_executable(conv2d_gpu conv2d_gpu.cpp)
target_link_libraries(conv2d_gpu PRIVATE Halide::Halide Halide::ImageIO Halide::Tools)

add_executable(depthwise_conv2d_gpu depthwise_conv2d_gpu.cpp)
target_link_libraries(depthwise_conv2d_gpu PRIVATE Halide::Halide Halide::ImageIO Halide::Tools)

add_executable(maxpool_gpu maxpool_gpu.cpp)
target_link_libraries(maxpool_gpu PRIVATE Halide::Halide Halide::ImageIO Halide::Tools)
