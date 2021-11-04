cmake_minimum_required(VERSION 3.16)
project(kernel)

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

add_library(kernel
            ${SOURCE_FILES}
            ${HEADER_FILES})
target_link_libraries(kernel PRIVATE Halide::Halide)
