cmake_minimum_required(VERSION 3.16)
project(HaNNA)

#enable_testing()
# Set up language settings
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED YES)
set(CMAKE_CXX_EXTENSIONS NO)

find_package(Halide REQUIRED)

add_executable(conv2d conv2d.cpp)
target_link_libraries(conv2d PRIVATE Halide::Halide Halide::ImageIO)

add_executable(depth_separable_conv2d depth_separable_conv2d.cpp)
target_link_libraries(depth_separable_conv2d PRIVATE Halide::Halide Halide::ImageIO)
