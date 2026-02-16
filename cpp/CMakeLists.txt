cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(pyraformer_train)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

include_directories(include)

# Gather source files
file(GLOB SOURCES "src/*.cpp")

add_executable(pyraformer_train ${SOURCES})
target_link_libraries(pyraformer_train "${TORCH_LIBRARIES}")
set_property(TARGET pyraformer_train PROPERTY CXX_STANDARD 17)
