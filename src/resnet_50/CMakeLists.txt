cmake_minimum_required(VERSION 3.16)
project(resnet_50)

#enable_testing()


# Set up language settings
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED YES)
set(CMAKE_CXX_EXTENSIONS NO)

# Find Halide
find_package(Halide REQUIRED)

# Generator
add_executable(resnet_50.generator resnet_50_generator.cpp)
include_directories(${kernel_SOURCE_DIR})
target_link_libraries(resnet_50.generator
                      PRIVATE
                      kernel
                      Halide::Generator)

# Filters
add_halide_library(resnet_50 FROM resnet_50.generator
		   GENERATOR resnet_50)

# Main executable
add_executable(resnet_50_process process.cpp)
target_link_libraries(resnet_50_process
                      PRIVATE
                      Halide::ImageIO
                      resnet_50)

# Test that the app actually works!
add_test(NAME resnet_50_process COMMAND resnet_50_process)
set_tests_properties(resnet_50_process PROPERTIES
                     LABELS resnet_50
                     PASS_REGULAR_EXPRESSION "Success!"
                     SKIP_REGULAR_EXPRESSION "\\[SKIP\\]")