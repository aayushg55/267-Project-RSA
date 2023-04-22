# cmake_minimum_required(VERSION 3.14)
# project(hw2_2)
# enable_language(CUDA)
# set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address -g")

# set(CMAKE_CXX_STANDARD 14)
# set(CMAKE_CXX_STANDARD_REQUIRED YES)

# # find_package(MPI REQUIRED)

# # Group number
# set(GROUP_NAME "None" CACHE STRING "Your group name as it appears on bCourses (no spaces)")

# # Packaging system
# if (NOT ${GROUP_NAME} STREQUAL None)
#     set(CPACK_GENERATOR TGZ)
#     set(CPACK_PACKAGE_FILE_NAME "cs267${GROUP_NAME}_hw2_2")
#     install(FILES rsa_mpi.cu DESTINATION .)
#     install(FILES ${CPACK_PACKAGE_FILE_NAME}.pdf DESTINATION .)
#     include(CPack)
# endif ()

# # # We require the GNU compiler for this assignment.
# # if (NOT "${CMAKE_C_COMPILER_ID}" STREQUAL "GNU")
# #     set(Prg-Clang "PrgEnv-cray")
# #     message(WARNING
# #             "Must use GNU Compiler for submission. Make sure you ran:\n"
# #             "module swap ${Prg-${CMAKE_C_COMPILER_ID}} PrgEnv-gnu")
# # endif ()

# add_executable(mpi main.cu rsa_mpi.cu)
# # target_link_libraries(mpi PRIVATE MPI::MPI_CXX)
# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_80")
# set_target_properties(mpi PROPERTIES CUDA_ARCHITECTURES "80")

# # Copy the job scripts
# configure_file(job-mpi job-mpi COPYONLY)
