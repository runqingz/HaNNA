cmake_minimum_required(VERSION 3.16)
project(HaNNA)

#enable_testing()
# Set up language settings
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED YES)
set(CMAKE_CXX_EXTENSIONS NO)


add_subdirectory(kernel)
add_subdirectory(resnet_50)