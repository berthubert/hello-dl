cmake_minimum_required(VERSION 3.1)
set(CMAKE_CXX_FLAGS "-Wall -Wextra -Og -g -march=native -msse2")

find_package (Eigen3 3.3 REQUIRED NO_MODULE)
 
project(hello-dl VERSION 1.0
                  DESCRIPTION "Hello deep learning"
                  LANGUAGES CXX)


set(CMAKE_CXX_STANDARD 17 CACHE STRING "The C++ standard to use")
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)

set(CMAKE_THREAD_PREFER_PTHREAD TRUE)
set(THREADS_PREFER_PTHREAD_FLAG TRUE)
find_package(Threads REQUIRED)

add_executable(hello-dl hello-dl.cc mnistreader.cc)
target_link_libraries(hello-dl
  Eigen3::Eigen
  z
)

add_executable(tensor tensor.cc mnistreader.cc)
target_link_libraries(tensor
  Eigen3::Eigen
  z
)

