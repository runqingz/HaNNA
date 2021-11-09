cmake_minimum_required(VERSION 3.16)
project(kernels)

#enable_testing()
# Set up language settings
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED YES)
set(CMAKE_CXX_EXTENSIONS NO)
list(APPEND CMAKE_PREFIX_PATH "E:/dev/vcpkg/installed/x64-windows")

find_package(Halide REQUIRED)

add_executable(conv2d_gpu conv2d_gpu.cpp)
target_link_libraries(conv2d_gpu PRIVATE Halide::Halide Halide::ImageIO Halide::Tools)

add_executable(depthwise_conv2d_gpu depthwise_conv2d_gpu.cpp)
target_link_libraries(depthwise_conv2d_gpu PRIVATE Halide::Halide Halide::ImageIO Halide::Tools)

add_executable(relu_gpu relu_gpu.cpp)
target_link_libraries(relu_gpu PRIVATE Halide::Halide Halide::ImageIO Halide::Tools)

