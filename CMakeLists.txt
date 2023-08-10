cmake_minimum_required(VERSION 3.14.0 FATAL_ERROR)
project(runtimeDL)

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")

find_package(OpenCV REQUIRED)
find_package(Torch REQUIRED)
find_package(TorchVision REQUIRED)

option(WITH_ALGORITHMS "with sample" ON)
if(WITH_ALGORITHMS)
   add_subdirectory(algorithms)
endif()

option(WITH_SAMPLE "with sample" ON)
if(WITH_SAMPLE)
   add_subdirectory(samples)
endif()


