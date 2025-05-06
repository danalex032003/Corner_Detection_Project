cmake_minimum_required(VERSION 3.30)
project(Corner_Detection_Project)

set(CMAKE_CXX_STANDARD 20)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(Corner_Detection_Project main.cpp
    src/image_manipulation.h
    src/image_manipulation.cpp
    src/ShiTomasi.h
    src/ShiTomasi.cpp
)
target_link_libraries(Corner_Detection_Project ${OpenCV_LIBS})