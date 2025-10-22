cmake_minimum_required(VERSION 3.23.1 FATAL_ERROR)

project(ImMeshRender LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)
include(cmake/CPM.cmake)


option(ENABLE_HALF_PRECISION_TRANING_ "enable half-precision traning" OFF)

if(ENABLE_HALF_PRECISION_TRANING_)
    add_definitions(-DENABLE_HALF_PRECISION_TRANING)
endif(ENABLE_HALF_PRECISION_TRANING_)

option(ENABLE_CUDA_DEBUG_PRINT_ "enable cuda debug print" ON)

if(ENABLE_CUDA_DEBUG_PRINT_)
    add_definitions(-DENABLE_CUDA_DEBUG_PRINT)
endif(ENABLE_CUDA_DEBUG_PRINT_)
set(CMAKE_VERBOSE_MAKEFILE ON)


SET(CMAKE_BUILD_TYPE "Release")

find_package(Eigen3 REQUIRED)
find_package(OpenGL REQUIRED)
find_package(GLEW REQUIRED)
find_library(EGL_LIBRARY EGL REQUIRED)

find_package(OpenCV REQUIRED)

set (Torch_DIR /root/libtorch/libtorch/share/cmake/Torch)
find_package(Torch REQUIRED)
find_package(Pangolin REQUIRED)
find_package(OpenMP REQUIRED)
find_package(TBB REQUIRED)

add_subdirectory(third_party/cuCollections)

set (cuco_DIR ./build/third_party/cuCollections)
find_package(cuco REQUIRED)


set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}  ${TORCH_C_FLAGS} ${OpenMP_C_FLAGS}")

add_subdirectory(third_party/ReplicaSDK)

find_package(glm CONFIG REQUIRED)

include_directories(
    ${CUDA_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${OpenMP_INCLUDE_DIRS}
    cuda
    ${EIGEN3_INCLUDE_DIRS}
    ${TORCH_INCLUDE_DIRS}
    ${Pangolin_INCLUDE_DIRS}
    /root/tensorboardCpp
    /root/.cache/bazel/_bazel_root/64df53cb89265c6bbbf95698f7ffadcd/execroot/tensorboard.cpp/bazel-out/k8-fastbuild/bin
    /root/.cache/bazel/_bazel_root/64df53cb89265c6bbbf95698f7ffadcd/external/com_google_protobuf/src
    ./third_party/tensorboardCpp
    ./
    ${TBB_INCLUDE_DIRS}
    ${OPENGL_INCLUDE_DIR}
    ${GLEW_INCLUDE_DIRS}
)



add_executable(render
    src/MeshLearner.cu
)

# -rdc=true
target_compile_options(render PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
                       --default-stream per-thread
                       -Xptxas -v
                       -O3
                    #    -g
                       >) # -g -G

target_link_libraries(render
    ReplicaSDK
    -lpthread
    -lm
    glm::glm
    ${TORCH_LIBRARIES}
    ${CMAKE_DL_LIBS}
    ${OpenCV_LIBS}
    ${Pangolin_LIBRARIES}
    ${OpenMP_LIBRARIES}
    -lstdc++fs
    /root/tensorboardCpp/libtensorboardCppWan.so
    TBB::tbb
    cuco
    GL
    GLEW
    -lEGL
    ${EGL_LIBRARY}
)