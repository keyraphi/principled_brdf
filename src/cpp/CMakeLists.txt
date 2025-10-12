# Find required packages
find_package(CUDAToolkit REQUIRED)

# Include nanobind as a subdirectory (assuming we have nanobind in the project or we use FetchContent)
include(FetchContent)
FetchContent_Declare(
    nanobind
    GIT_REPOSITORY https://github.com/wjakob/nanobind
    GIT_TAG        master
)
FetchContent_MakeAvailable(nanobind)

# We'll create two libraries: one for CPU and one for CUDA, but we'll combine them into one module.

# CPU library
add_library(disney_brdf_cpu STATIC
    cpu/disney_brdf_cpu.cpp
)

target_include_directories(disney_brdf_cpu PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# CUDA library
enable_language(CUDA)
add_library(disney_brdf_cuda STATIC
    cuda/disney_brdf_cuda.cu
)

target_include_directories(disney_brdf_cuda PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(disney_brdf_cuda PRIVATE CUDA::cudart)

# Now, the main module that uses both CPU and CUDA
nanobind_add_module(disney_brdf_core
    disney_brdf.cpp
)

target_link_libraries(disney_brdf_core PRIVATE disney_brdf_cpu disney_brdf_cuda)

# We need to link against CUDA if we are using CUDA code
target_link_libraries(disney_brdf_core PRIVATE CUDA::cudart)

# We also need to set the C++ standard for the module
target_compile_features(disney_brdf_core PRIVATE cxx_std_17)

# If we are on a non-Windows platform, we need to set the rpath
if(UNIX)
    set_target_properties(disney_brdf_core PROPERTIES INSTALL_RPATH "$ORIGIN")
endif()
