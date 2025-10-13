# Test configuration for principled_brdf
enable_testing()

# Find Python for testing
find_package(Python REQUIRED COMPONENTS Interpreter)

# Copy test scripts to build directory
set(TEST_SCRIPTS
    test_basic_functionality.py
    test_consistency.py
)

foreach(script ${TEST_SCRIPTS})
    configure_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/${script}
        ${CMAKE_CURRENT_BINARY_DIR}/${script}
        COPYONLY
    )
endforeach()

# Test 1: Basic functionality test
add_test(
    NAME test_basic_functionality
    COMMAND ${Python_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/test_basic_functionality.py
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)

# Test 2: Consistency test (CPU vs GPU vs PyTorch)
add_test(
    NAME test_consistency
    COMMAND ${Python_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/test_consistency.py
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)

# Test 3: Gradient test (CPU vs GPU vs PyTorch autograd)
add_test(
    NAME test_gradients
    COMMAND ${Python_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/test_consistency.py
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)

# Set test labels for organizing tests
set_tests_properties(test_basic_functionality PROPERTIES LABELS "quick")
set_tests_properties(test_consistency PROPERTIES LABELS "functionality")

# Simple custom target to run all tests
add_custom_target(run_tests
    COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure
    COMMENT "Running all tests..."
)

# Print test summary
message(STATUS "")
message(STATUS "Test Configuration:")
message(STATUS "  Basic functionality test: YES")
message(STATUS "  Consistency test: YES")
message(STATUS "")
message(STATUS "Run 'ninja run_tests' to execute all tests")
