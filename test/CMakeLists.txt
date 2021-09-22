set(TEST_SRC 
  test_linspace_vector_gen.cpp
  test_pq.cpp
  test_ift.cpp
  test_geodesic_centers.cpp)

set(MAIN_SRC
    test_ift_performance.cpp)

add_executable(lp_test ${TEST_SRC})

add_executable(perf_test ${MAIN_SRC})

include(FetchContent)
FetchContent_Declare(googletest
  URL "https://github.com/google/googletest/archive/release-1.10.0.tar.gz")

FetchContent_MakeAvailable(googletest)

find_package(OpenMP REQUIRED)
find_package(BLAS REQUIRED)

target_link_libraries(perf_test PRIVATE label_propagation ${BLAS_LIBRARIES} OpenMP::OpenMP_CXX)

target_include_directories(lp_test PRIVATE ${IFT_INCLUDE_DIRS})

target_link_libraries(lp_test PRIVATE
label_propagation
${BLAS_LIBRARIES}
OpenMP::OpenMP_CXX
gtest_main
)

# Defines `gtest_discover_tests()`.
include(GoogleTest)
gtest_discover_tests(lp_test)