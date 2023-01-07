cmake_minimum_required(VERSION 3.1)
set(CMAKE_CXX_FLAGS "-Wall -Wextra -O3 -ggdb -march=native -msse2 -mavx -mavx2  -fno-omit-frame-pointer -ffast-math -Iext/imgui -Iext/imgui/backends -Iext/implot-master")

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



add_library(common OBJECT mnistreader.cc textsupport.cc)

add_executable(tensor tensor.cc)
target_link_libraries(tensor z  common)

add_executable(first-relu first-relu.cc )
target_link_libraries(first-relu z  common)

add_executable(first-convo first-convo.cc )
target_link_libraries(first-convo z  common)

add_executable(modular-convo modular-convo.cc )
target_link_libraries(modular-convo z  common)

add_executable(modular-linear-convo modular-linear-convo.cc )
target_link_libraries(modular-linear-convo z  common)


add_executable(worker-convo worker-convo.cc  )
target_link_libraries(worker-convo  z  common)

add_executable(worker-convo-avx worker-convo-avx.cc )
target_link_libraries(worker-convo-avx  z  common)

add_executable(worker-convo-avx-threaded worker-convo-avx-threaded.cc )
target_link_libraries(worker-convo-avx-threaded z  common)


add_executable(modular-threaded-convo modular-threaded-convo.cc )
target_link_libraries(modular-threaded-convo z  common)


add_executable(try-convo try-convo.cc )
target_link_libraries(try-convo z  common)


add_executable(los los.cc )
target_link_libraries(los z  common)

add_executable(los2 los2.cc )
target_link_libraries(los2 z  common)

add_executable(los3 los3.cc )
target_link_libraries(los3 z  common)



add_executable(testrunner testrunner.cc arraytests.cc testfvector.cc grutest.cc workertest.cc)

target_link_libraries(testrunner  z  common)

enable_testing()
add_test(testname testrunner)
