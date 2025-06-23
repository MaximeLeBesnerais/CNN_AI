cmake_minimum_required(VERSION 3.16)
project(CNN_Framework)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Compiler flags for optimization and debugging
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -march=native")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -Wall -Wextra")

# Include directories
include_directories(include)

# Source files
set(SOURCES
    src/tensor.cpp
    src/linear.cpp
    src/activation.cpp
    src/conv2d.cpp
    src/pooling.cpp
    src/dropout.cpp
    src/layers.cpp
    src/blocks.cpp
    src/model.cpp
    src/optimizer.cpp
    src/loss.cpp
)

# Header files
file(GLOB_RECURSE HEADERS "include/*.h")

# Create library
add_library(cnn_framework STATIC ${SOURCES} ${HEADERS})

# Find required packages
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(cnn_framework PUBLIC OpenMP::OpenMP_CXX)
endif()

# Example executables
add_executable(cnn_example examples/main.cpp)
target_link_libraries(cnn_example cnn_framework)

add_executable(cnn_conv_example examples/cnn_example.cpp)
target_link_libraries(cnn_conv_example cnn_framework)

add_executable(cnn_advanced_blocks examples/advanced_blocks_example.cpp)
target_link_libraries(cnn_advanced_blocks cnn_framework)

add_executable(test_blocks_compile examples/test_blocks_compile.cpp)
target_link_libraries(test_blocks_compile cnn_framework)

# Test executable
add_executable(cnn_tests tests/test_main.cpp tests/test_tensor.cpp tests/test_layers.cpp tests/test_advanced_blocks.cpp)
target_link_libraries(cnn_tests cnn_framework)

# Enable testing
enable_testing()
add_test(NAME TensorTests COMMAND cnn_tests)
