cmake_minimum_required(VERSION 2.8)


find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)


cuda_add_executable(voronoi_demo demo.cpp)