set(SRC 
    label_propagation.cpp
    utils.cpp
    priority_queue.cpp
    ift.cpp)

set(HEADERS
    label_propagation.h
    utils.h
    priority_queue.h
    ift.h)

set(HEADERS ${HEADERS} PARENT_SCOPE)

add_library(label_propagation SHARED ${SRC})
target_include_directories(label_propagation PUBLIC $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>)
set_target_properties(label_propagation
                      PROPERTIES PUBLIC_HEADER ${PROJECT_SOURCE_DIR}/label_propagation/ift.h)

find_package(OpenMP REQUIRED)
find_package(BLAS REQUIRED)

target_link_libraries(label_propagation PRIVATE ${BLAS_LIBRARIES} OpenMP::OpenMP_CXX)

install(TARGETS label_propagation
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/label_propagation)