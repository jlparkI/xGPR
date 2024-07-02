cmake_minimum_required(VERSION 3.15)
project(${SKBUILD_PROJECT_NAME} LANGUAGES CXX)

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


if (CMAKE_VERSION VERSION_LESS 3.18)
  set(DEV_MODULE Development)
else()
  set(DEV_MODULE Development.Module)
endif()

find_package(Python 3.8 COMPONENTS Interpreter ${DEV_MODULE} REQUIRED)

if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/ext/nanobind)
find_package(nanobind CONFIG REQUIRED)

nanobind_add_module(
  xgpr_cpu_rfgen_cpp_ext

  # Target the stable ABI for Python 3.12+, which reduces
  # the number of binary wheels that must be built. This
  # does nothing on older Python versions
  STABLE_ABI

  NB_STATIC

  xGPR/random_feature_generation/cpu_rf_gen/xgpr_cpu_rfgen_cpp_ext.cpp
  xGPR/random_feature_generation/cpu_rf_gen/shared_fht_functions/hadamard_transforms.cpp
  xGPR/random_feature_generation/cpu_rf_gen/shared_fht_functions/shared_rfgen_ops.cpp
  xGPR/random_feature_generation/cpu_rf_gen/shared_fht_functions/simplex_rff_projections.cpp
  xGPR/random_feature_generation/cpu_rf_gen/basic_ops/transform_functions.cpp
  xGPR/random_feature_generation/cpu_rf_gen/rbf_ops/rbf_ops.cpp
  xGPR/random_feature_generation/cpu_rf_gen/rbf_ops/ard_ops.cpp
  xGPR/random_feature_generation/cpu_rf_gen/convolution_ops/conv1d_operations.cpp
  xGPR/random_feature_generation/cpu_rf_gen/convolution_ops/rbf_convolution.cpp

)

install(TARGETS xgpr_cpu_rfgen_cpp_ext LIBRARY DESTINATION ${SKBUILD_PROJECT_NAME})
