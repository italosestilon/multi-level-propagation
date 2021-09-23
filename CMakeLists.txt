cmake_minimum_required(VERSION 3.17)
project(label-propagation VERSION 0.1.0
    LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 11)

add_compile_options(-Wall -Wextra -pedantic -Werror)

message(STATUS "CXX_STANDARD: ${CMAKE_CXX_STANDARD}")
message(STATUS "CXX_FLAGS: ${CMAKE_CXX_FLAGS}")
message(STATUS "CMAKE_CXX_COMPILER: ${CMAKE_CXX_COMPILER}")
message(STATUS "CMAKE_SHARED_LINKER: ${CMAKE_SHARED_LINKER_FLAGS}")

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/")

include(CTest)
enable_testing()

add_subdirectory(label_propagation)
add_subdirectory(test)
add_subdirectory(python)

# add_library(label-propagation label-propagation.cpp)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
