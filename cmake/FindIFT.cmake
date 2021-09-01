include(FindPackageHandleStandardArgs)

find_library(IFT_LIBRARY
    NAMES ift
    PATHS ${LIBIFT_ROOT}/lib
    NO_DEFAULT_PATH)

message(STATUS "LIBIFT_ROOT: ${LIBIFT_ROOT}/lib")
message(STATUS "IFT_LIBRARY: ${IFT_LIBRARY}")
    
# SET(IFT_LIBRARIES)
set(IFT_INCLUDE_DIRS ${LIBIFT_ROOT}/include)

file(GLOB IFT_EXTERNAL
        ${LIBIFT_ROOT}/externals/*/)

foreach(lib ${IFT_EXTERNAL})
    if(EXISTS ${lib}/src)
        message(STATUS ${lib};)
        list(APPEND IFT_INCLUDE_DIRS ${lib}/include)
    endif()
endforeach()

find_package_handle_standard_args(IFT REQUIRED_VARS IFT_LIBRARY IFT_INCLUDE_DIRS)