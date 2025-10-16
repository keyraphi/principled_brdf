# Create an interface library for common utilities
add_library(common_utils INTERFACE)
target_include_directories(common_utils INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
