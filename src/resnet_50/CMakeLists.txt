cmake_minimum_required(VERSION 3.16)
project(resnet50)

#enable_testing()


# Set up language settings
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED YES)
set(CMAKE_CXX_EXTENSIONS NO)

# Find Halide
find_package(Halide REQUIRED)

# Generator
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fno-rtti")
add_executable(resnet50.generator resnet_50_generator.cpp)
include_directories(${kernel_SOURCE_DIR})
target_link_libraries(resnet50.generator
                      PRIVATE
                      Halide::Generator)

# Filters
add_halide_library(resnet50 FROM resnet50.generator
		   GENERATOR resnet50
           FEATURES metal)

# Main executable
add_executable(resnet_50_process process.cpp)
target_link_libraries(resnet_50_process
                      PRIVATE
                      Halide::ImageIO
                      resnet50)

# Test that the app actually works!
add_test(NAME resnet_50_process COMMAND resnet_50_process)
set_tests_properties(resnet_50_process PROPERTIES
                     LABELS resnet50
                     PASS_REGULAR_EXPRESSION "Success!"
                     SKIP_REGULAR_EXPRESSION "\\[SKIP\\]")