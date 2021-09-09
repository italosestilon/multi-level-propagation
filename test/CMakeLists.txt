set(TEST_SRC 
  test_linspace_vector_gen.cpp
  test_pq.cpp
  test_ift.cpp)

add_executable(lp_test ${TEST_SRC})

include(FetchContent)
FetchContent_Declare(googletest
  URL "https://github.com/google/googletest/archive/release-1.10.0.tar.gz")

FetchContent_MakeAvailable(googletest)

target_link_libraries(lp_test PRIVATE label_propagation)

find_package(OpenMP REQUIRED)

target_include_directories(lp_test PRIVATE ${IFT_INCLUDE_DIRS})

target_link_libraries(lp_test PRIVATE
OpenMP::OpenMP_CXX
gtest_main
)

# Defines `gtest_discover_tests()`.
include(GoogleTest)
gtest_discover_tests(lp_test)