cmake_minimum_required(VERSION 3.1)
set(CMAKE_CXX_FLAGS "-Wall -Wextra -O3 -ggdb -march=native -msse2 -mavx -mavx2")

project(hello-dl VERSION 1.0
                  DESCRIPTION "Hello deep learning"
                  LANGUAGES CXX)


set(CMAKE_CXX_STANDARD 17 CACHE STRING "The C++ standard to use")
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)

set(CMAKE_THREAD_PREFER_PTHREAD TRUE)
set(THREADS_PREFER_PTHREAD_FLAG TRUE)
find_package(Threads REQUIRED)

#add_executable(hello-dl hello-dl.cc mnistreader.cc)
#target_link_libraries(hello-dl
#  z
#)

add_executable(tensor tensor.cc mnistreader.cc)
target_link_libraries(tensor
  z
)

add_executable(first-relu first-relu.cc mnistreader.cc)
target_link_libraries(first-relu
  z
)

add_executable(first-convo first-convo.cc mnistreader.cc)
target_link_libraries(first-convo
  z
)

add_executable(modular-convo modular-convo.cc mnistreader.cc)
target_link_libraries(modular-convo
  z
)

add_executable(modular-linear-convo modular-linear-convo.cc mnistreader.cc)
target_link_libraries(modular-linear-convo
  z
)

add_executable(modular-threaded-convo modular-threaded-convo.cc mnistreader.cc)
target_link_libraries(modular-threaded-convo
  z
)


add_executable(try-convo try-convo.cc mnistreader.cc)
target_link_libraries(try-convo
  z
)


add_executable(los los.cc)
target_link_libraries(los
  z
)

add_executable(testrunner testrunner.cc arraytests.cc testfvector.cc)

target_link_libraries(testrunner
  z
)

enable_testing()
add_test(testname testrunner)
