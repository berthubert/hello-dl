cmake_minimum_required(VERSION 3.1)
set(CMAKE_CXX_FLAGS "-Wall -Wextra -O3 -ggdb -march=native -msse2 -mavx -mavx2 -mfma  -fno-omit-frame-pointer -ffast-math -Iext/imgui -Iext/imgui/backends -Iext/implot-master -I/usr/include/eigen3 -I/usr/include/eigen3/unsupported/ -Wno-missing-field-initializers")

project(hello-dl VERSION 1.0
                  DESCRIPTION "Hello deep learning"
                  LANGUAGES CXX)


#add_compile_options(-fsanitize=memory)
#add_link_options(-fsanitize=memory)
                
set(CMAKE_CXX_STANDARD 17 CACHE STRING "The C++ standard to use")
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)

set(CMAKE_THREAD_PREFER_PTHREAD TRUE)
set(THREADS_PREFER_PTHREAD_FLAG TRUE)
find_package(Threads REQUIRED)
find_package (Eigen3 3.3 REQUIRED NO_MODULE)

add_library(common OBJECT mnistreader.cc textsupport.cc )
#add_library(gui OBJECT ui.cc ext/imgui/imgui.cpp  ext/imgui/imgui_demo.cpp  ext/imgui/imgui_draw.cpp  ext/imgui/imgui_tables.cpp  ext/imgui/imgui_widgets.cpp ext/imgui/backends/imgui_impl_glfw.cpp ext/imgui/backends/imgui_impl_opengl3.cpp ext/implot-master/implot.cpp ext/implot-master/implot_items.cpp ext/implot-master/implot_demo.cpp)



add_executable(tensor-relu tensor-relu.cc ext/sqlitewriter/sqlwriter.cc)
target_link_libraries(tensor-relu z  common Eigen3::Eigen sqlite3 Threads::Threads)

add_executable(tensor-convo tensor-convo.cc ext/sqlitewriter/sqlwriter.cc )
target_link_libraries(tensor-convo z  common Eigen3::Eigen sqlite3 Threads::Threads) 

add_executable(tensor-convo-par tensor-convo-par.cc ext/sqlitewriter/sqlwriter.cc )
target_link_libraries(tensor-convo-par z common Eigen3::Eigen sqlite3 Threads::Threads) 


add_executable(img-ocr img-ocr.cc vizi.cc )
target_link_libraries(img-ocr z  common Eigen3::Eigen Threads::Threads) 

add_executable(threeorseven threeorseven.cc vizi.cc ext/sqlitewriter/sqlwriter.cc)
target_link_libraries(threeorseven z  common Eigen3::Eigen sqlite3 Threads::Threads) 

add_executable(37learn 37learn.cc vizi.cc ext/sqlitewriter/sqlwriter.cc)
target_link_libraries(37learn z  common Eigen3::Eigen sqlite3 Threads::Threads) 

add_executable(basic-autograd basic-autograd.cc)
target_link_libraries(basic-autograd z  common Eigen3::Eigen Threads::Threads ) 

add_executable(imagine imagine.cc vizi.cc)
target_link_libraries(imagine z  common Eigen3::Eigen Threads::Threads ) 


add_executable(tensor-gru tensor-gru.cc )
target_link_libraries(tensor-gru z  common Eigen3::Eigen Threads::Threads )

add_executable(testrunner testrunner.cc arraytests.cc testfvector.cc grutest.cc tensor2test.cc tensormodtest.cc)

target_link_libraries(testrunner  z  common Eigen3::Eigen)

enable_testing()
add_test(testname testrunner)



#add_executable(first-convo first-convo.cc )
#target_link_libraries(first-convo z  common)

#add_executable(modular-convo modular-convo.cc )
#target_link_libraries(modular-convo z  common)

#add_executable(modular-linear-convo modular-linear-convo.cc )
#target_link_libraries(modular-linear-convo z  common)


#add_executable(worker-convo worker-convo.cc  )
#target_link_libraries(worker-convo  z  common)

#add_executable(worker-convo-avx worker-convo-avx.cc)
#target_link_libraries(worker-convo-avx  z  common gui GL glfw)

#add_executable(worker-convo-avx-threaded worker-convo-avx-threaded.cc )
#target_link_libraries(worker-convo-avx-threaded z  common)


#add_executable(modular-threaded-convo modular-threaded-convo.cc )
#target_link_libraries(modular-threaded-convo z  common)


add_executable(try-convo try-convo.cc vizi.cc)
target_link_libraries(try-convo z  common Threads::Threads)


#add_executable(los los.cc )
#target_link_libraries(los z  common)

add_executable(mnistposter mnistposter.cc )
target_link_libraries(mnistposter z  common)

#add_executable(los2 los2.cc )
#target_link_libraries(los2 z  common)

#add_executable(los3 los3.cc )
#target_link_libraries(los3 z  common)

#add_executable(tensor tensor.cc)
#target_link_libraries(tensor z  common)

#add_executable(first-relu first-relu.cc )
#target_link_libraries(first-relu z  common)


