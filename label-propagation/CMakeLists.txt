set(SRC label_propagation.cpp)
set(HEADERS label_propagation.h)

set(HEADERS ${HEADERS} PARENT_SCOPE)

message(STATUS "HEADERS " ${HEADERS})

add_library(label_propagation ${SRC})
target_include_directories(label_propagation PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

find_package(OpenMP REQUIRED)
target_link_libraries(label_propagation PRIVATE OpenMP::OpenMP_CXX)

find_package(IFT REQUIRED)
target_link_libraries(label_propagation PUBLIC ${IFT_LIBRARY})
target_include_directories(label_propagation PRIVATE ${IFT_INCLUDE_DIRS})