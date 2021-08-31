cmake_minimum_required(VERSION 3.0 FATAL_ERROR)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

project(midas-test)

find_package(Torch REQUIRED)
FIND_PACKAGE( OpenCV  REQUIRED )


add_executable(midas-test inference_midas.cpp midas.cpp)
target_link_libraries(midas-test torch)
target_link_libraries(midas-test ${OpenCV_LIBRARIES})
