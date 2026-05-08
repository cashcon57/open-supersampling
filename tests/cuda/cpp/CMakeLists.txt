cmake_minimum_required(VERSION 3.24)
if(DEFINED ENV{CXX})
  set(CMAKE_CUDA_HOST_COMPILER "$ENV{CXX}" CACHE FILEPATH "CUDA host compiler")
endif()
project(oss_cuda_gtests LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 80 86 89)
endif()

get_filename_component(REPO_ROOT "${CMAKE_CURRENT_LIST_DIR}/../../.." ABSOLUTE)
set(OSS_CUDA_SRC "${REPO_ROOT}/oss/cuda/src")
set(OSS_CUDA_EXTRA_INCLUDES "")
if(DEFINED ENV{CUDA_HOME} AND EXISTS "$ENV{CUDA_HOME}/targets/x86_64-linux/include")
  list(APPEND OSS_CUDA_EXTRA_INCLUDES "$ENV{CUDA_HOME}/targets/x86_64-linux/include")
endif()
if(DEFINED ENV{CUDA_HOME} AND EXISTS "$ENV{CUDA_HOME}/targets/x86_64-linux/include/cccl")
  list(APPEND OSS_CUDA_EXTRA_INCLUDES "$ENV{CUDA_HOME}/targets/x86_64-linux/include/cccl")
endif()

find_package(CUDAToolkit REQUIRED)

include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
)
FetchContent_MakeAvailable(googletest)

add_executable(test_oss_cuda
  test_preprocess.cu
  test_build_pairs.cu
  "${OSS_CUDA_SRC}/rasterizer_fwd.cu"
)

target_compile_definitions(test_oss_cuda PRIVATE OSS_CUDA_KERNELS_ONLY)
target_include_directories(test_oss_cuda PRIVATE
  "${OSS_CUDA_SRC}" ${OSS_CUDA_EXTRA_INCLUDES}
)
target_compile_options(test_oss_cuda PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr --expt-extended-lambda -lineinfo>
)
target_link_libraries(test_oss_cuda PRIVATE GTest::gtest_main CUDA::cudart)
set_target_properties(test_oss_cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

include(GoogleTest)
gtest_discover_tests(test_oss_cuda)
