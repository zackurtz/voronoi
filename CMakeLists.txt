cmake_minimum_required(VERSION 2.8)

set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} "--std=c++11")


find_package(OpenCV 3 REQUIRED)
find_package(CUDA REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${CUDA_INCLUDE_DIRS})

message("OpenCV_INCLUDE_DIRS ${OpenCV_INCLUDE_DIRS}")


cuda_add_executable(voronoi_demo demo.cpp voronoi.cu)
target_link_libraries(voronoi_demo ${OpenCV_LIBRARIES})