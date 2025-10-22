cmake_minimum_required(VERSION 3.23.1 FATAL_ERROR)
project(ReplicaSDK LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
# include(../cmake/CPM.cmake)

# CPMAddPackage(
#     NAME cuco
#     GITHUB_REPOSITORY NVIDIA/cuCollections
#     GIT_TAG dev
#     OPTIONS
#         "BUILD_TESTS OFF"
#         "BUILD_BENCHMARKS OFF"
#         "BUILD_EXAMPLES OFF"
# )

set (cuco_DIR ../../build/third_party/cuCollections)
find_package(cuco REQUIRED)

file(GLOB_RECURSE CURRENT_HEADERS  *.h *.hpp *.cuh)
file(GLOB_RECURSE CURRENT_SOURCES  *.cpp *.c *.cu)

find_package(Eigen3 REQUIRED)

include_directories(
./
${EIGEN3_INCLUDE_DIRS}
)


# set(CMAKE_CUDA_FLAGS ${CMAKE_CUDA_FLAGS} "-g -G -lineinfo -std=c++17")
# set(CMAKE_CUDA_FLAGS ${CMAKE_CUDA_FLAGS} "-O3")

source_group("Include" FILES ${CURRENT_HEADERS}) 
source_group("Source" FILES ${CURRENT_SOURCES}) 

add_library(ReplicaSDK SHARED ${CURRENT_SOURCES})

include_directories(
    ./
)

target_link_libraries(ReplicaSDK
    cuco
)

# target_compile_options(ReplicaSDK PRIVATE -G -Xptxas -v -O3)
target_compile_options(ReplicaSDK PRIVATE -O3)