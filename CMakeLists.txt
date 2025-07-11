include(CheckLanguage)

set(CMAKE_VERBOSE_MAKEFILE on)
cmake_minimum_required(VERSION 3.18)
if (NOT SKBUILD)
  message(WARNING "\
  This CMake file is meant to be executed using 'scikit-build'. Running
  it directly will almost certainly not produce the desired result. If
  you are a user trying to install this package, please use the command
  below, which will install all necessary build dependencies, compile
  the package in an isolated environment, and then install it.
  =====================================================================
   $ pip install .
  =====================================================================
  If you are a software developer, and this is your own package, then
  it is usually much more efficient to install the build dependencies
  in your environment once and use the following command that avoids
  a costly creation of a new virtual environment at every compilation:
  =====================================================================
   $ pip install nanobind scikit-build-core[pyproject]
   $ pip install --no-build-isolation -ve .
  =====================================================================
  You may optionally add -Ceditable.rebuild=true to auto-rebuild when
  the package is imported. Otherwise, you need to re-run the above
  after editing C++ files.")
endif()


check_language(CUDA)
if (CMAKE_CUDA_COMPILER)
    message(STATUS "CUDA is OK")
    include_directories(${CUDA_INCLUDE_DIRS})
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    add_compile_definitions(USE_CUDA)
    
    #set(CMAKE_CUDA_ARCHITECTURES 52)
    project(${SKBUILD_PROJECT_NAME} LANGUAGES CXX CUDA)
else()
    message(STATUS "No CUDA")
    remove_definitions(USE_CUDA)
    project(${SKBUILD_PROJECT_NAME} LANGUAGES CXX)
endif()

find_package(Python 3.8 COMPONENTS Interpreter Development.Module REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/ext/nanobind)

nanobind_add_module(
  xgpr_cpu_rfgen_cpp_ext

  # Target the stable ABI for Python 3.12+, which reduces
  # the number of binary wheels that must be built. This
  # does nothing on older Python versions
  #STABLE_ABI

  NB_STATIC

  ext/xgpr_private/random_feature_generation/cpu_rf_gen/xgpr_cpu_rfgen_cpp_ext.cpp
  ext/xgpr_private/random_feature_generation/cpu_rf_gen/shared_fht_functions/hadamard_transforms.cpp
  ext/xgpr_private/random_feature_generation/cpu_rf_gen/shared_fht_functions/shared_rfgen_ops.cpp
  ext/xgpr_private/random_feature_generation/cpu_rf_gen/basic_ops/transform_functions.cpp
  ext/xgpr_private/random_feature_generation/cpu_rf_gen/rbf_ops/rbf_ops.cpp
  ext/xgpr_private/random_feature_generation/cpu_rf_gen/rbf_ops/ard_ops.cpp
  ext/xgpr_private/random_feature_generation/cpu_rf_gen/convolution_ops/conv1d_operations.cpp
  ext/xgpr_private/random_feature_generation/cpu_rf_gen/convolution_ops/rbf_convolution.cpp
  ext/xgpr_private/random_feature_generation/cpu_rf_gen/classification_calcs/classification_calcs.cpp

)


if (CMAKE_CUDA_COMPILER)
    nanobind_add_module(
        xgpr_cuda_rfgen_cpp_ext

        # Target the stable ABI for Python 3.12+, which reduces
        # the number of binary wheels that must be built. This
        # does nothing on older Python versions
        #STABLE_ABI

        NB_STATIC

    ext/xgpr_private/random_feature_generation/gpu_rf_gen/xgpr_cuda_rfgen_cpp_ext.cpp
    ext/xgpr_private/random_feature_generation/gpu_rf_gen/basic_ops/basic_array_operations.cu
    ext/xgpr_private/random_feature_generation/gpu_rf_gen/rbf_ops/rbf_ops.cu
    ext/xgpr_private/random_feature_generation/gpu_rf_gen/rbf_ops/ard_ops.cu
    ext/xgpr_private/random_feature_generation/gpu_rf_gen/convolution_ops/convolution.cu
    ext/xgpr_private/random_feature_generation/gpu_rf_gen/convolution_ops/rbf_convolution.cu
    ext/xgpr_private/random_feature_generation/gpu_rf_gen/classification_calcs/classification_calcs.cu

    )
    install(TARGETS xgpr_cuda_rfgen_cpp_ext LIBRARY DESTINATION ${SKBUILD_PROJECT_NAME})
endif()

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(xgpr_cpu_rfgen_cpp_ext PUBLIC OpenMP::OpenMP_CXX)
endif()

install(TARGETS xgpr_cpu_rfgen_cpp_ext LIBRARY DESTINATION
    ${SKBUILD_PROJECT_NAME})
