set(SRC label_propagation.cpp utils.cpp)
set(HEADERS label_propagation.h utils.h)

set(HEADERS ${HEADERS} PARENT_SCOPE)

message(STATUS "HEADERS " ${HEADERS})

add_library(label_propagation ${SRC})
target_include_directories(label_propagation PUBLIC $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>)

find_package(OpenMP REQUIRED)
target_link_libraries(label_propagation PRIVATE OpenMP::OpenMP_CXX)

find_package(IFT REQUIRED)
target_link_libraries(label_propagation PRIVATE ${IFT_LIBRARY})
target_include_directories(label_propagation PRIVATE ${IFT_INCLUDE_DIRS})