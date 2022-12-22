cmake_minimum_required(VERSION 3.1)
set(CMAKE_CXX_FLAGS "-Wall -Wextra -O3 -g -march=native -msse2")

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

add_executable(first-relu first-relu.cc mnistreader.cc)
target_link_libraries(first-relu
  Eigen3::Eigen
  z
)

add_executable(first-convo first-convo.cc mnistreader.cc)
target_link_libraries(first-convo
  Eigen3::Eigen
  z
)


add_executable(los los.cc)
target_link_libraries(los
  Eigen3::Eigen
  z
)

add_executable(testrunner testrunner.cc arraytests.cc)

target_link_libraries(testrunner
  Eigen3::Eigen
  z
)

enable_testing()
add_test(testname testrunner)
