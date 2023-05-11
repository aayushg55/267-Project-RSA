cmake_minimum_required(VERSION 3.14)
project(hw3 LANGUAGES CXX)
enable_language(CUDA)
find_package(UPCXX REQUIRED)
# set(CMAKE_CXX_STANDARD 14)
# set(CMAKE_CXX_STANDARD_REQUIRED YES)

# Group number
set(GROUP_NAME "None" CACHE STRING "Your group name as it appears on bCourses (no spaces)")

# Packaging system
if (NOT ${GROUP_NAME} STREQUAL None)
    set(CPACK_GENERATOR TGZ)
    set(CPACK_PACKAGE_FILE_NAME "cs267${GROUP_NAME}_hw3")
    install(FILES main.cpp hash_map.hpp DESTINATION .)
    install(FILES ${CPACK_PACKAGE_FILE_NAME}.pdf DESTINATION .)
    include(CPack)
endif ()



# Build the kmer_hash executable
add_executable(main main.cpp)
target_link_libraries(main PRIVATE UPCXX::upcxx -lcublas -lcurand)
target_compile_features(main PRIVATE cxx_std_14)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_80")
set_target_properties(main PROPERTIES CUDA_ARCHITECTURES "80")
# Copy the job scripts
configure_file(job-perlmutter-starter job-perlmutter-starter COPYONLY)

