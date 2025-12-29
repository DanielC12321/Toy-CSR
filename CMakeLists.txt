cmake_minimum_required(VERSION 3.15)
project(my_sparse_matrix)
find_package(pybind11 REQUIRED)

pybind11_add_module(my_sparse wrapper.cpp)


if(MSVC)
    target_compile_options(my_sparse PRIVATE /O2)
else()
    target_compile_options(my_sparse PRIVATE -O3 -march=native)
endif()
find_package(OpenMP REQUIRED)
target_link_libraries(my_sparse PRIVATE OpenMP::OpenMP_CXX)