# CMake Minimum Version
cmake_minimum_required(VERSION 4.0.2)

# Project Definition
project(WebcamFilteringWithOpenCVandCUDANPP VERSION 0.1.0 LANGUAGES CXX)

# C++ Standard Configuration
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Output Directory Configuration
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/${CMAKE_CFG_INTDIR}/lib")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/${CMAKE_CFG_INTDIR}/lib")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/${CMAKE_CFG_INTDIR}/bin")

# CUDA Toolkit and NPP Configuration
find_package(CUDAToolkit REQUIRED COMPONENTS NPPIAL NPPICC NPPIF)

if(NOT CUDAToolkit_FOUND)
    message(FATAL_ERROR "CUDA Toolkit was not found, but is required.")
else()
    message(STATUS "CUDA Toolkit Found: Version ${CUDAToolkit_VERSION}")
    message(STATUS "  Libraries: ${CUDAToolkit_LIBRARIES}")
    message(STATUS "  Include Dirs: ${CUDAToolkit_INCLUDE_DIRS}")
endif()

# Visual Studio File Filters (Source Groups)
set(SOLUTION_ITEMS_FILES "")
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/README.md")
    list(APPEND SOLUTION_ITEMS_FILES "${CMAKE_CURRENT_SOURCE_DIR}/README.md")
endif()
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/.gitignore")
    list(APPEND SOLUTION_ITEMS_FILES "${CMAKE_CURRENT_SOURCE_DIR}/.gitignore")
endif()
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/vcpkg.json")
    list(APPEND SOLUTION_ITEMS_FILES "${CMAKE_CURRENT_SOURCE_DIR}/vcpkg.json")
endif()
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/vcpkg-configuration.json")
    list(APPEND SOLUTION_ITEMS_FILES "${CMAKE_CURRENT_SOURCE_DIR}/vcpkg-configuration.json")
endif()
# Add any other files you want here

# This target doesn't build anything; it's just a container in the VS Solution Explorer.
if(SOLUTION_ITEMS_FILES)
    add_custom_target(SolutionFiles SOURCES ${SOLUTION_ITEMS_FILES})

    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/README.md")
        source_group("Documentation" FILES "${CMAKE_CURRENT_SOURCE_DIR}/README.md")
    endif()
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/.gitignore")
        source_group("Version Control" FILES "${CMAKE_CURRENT_SOURCE_DIR}/.gitignore")
    endif()
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/vcpkg.json")
        source_group("Package Management" FILES "${CMAKE_CURRENT_SOURCE_DIR}/vcpkg.json")
    endif()
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/vcpkg-configuration.json")
        source_group("Package Management" FILES "${CMAKE_CURRENT_SOURCE_DIR}/vcpkg-configuration.json")
    endif()

    #    Place the "SolutionFiles" project into a solution folder named "SolutionFiles".
    #    May not work if the IDE is not VS or XCode
    set_target_properties(SolutionFiles PROPERTIES FOLDER "Solution Files")
else()
    message(STATUS "No solution-level miscellaneous files found to create SolutionItems target.")
endif()

# Process the CMakeLists.txt file located in the 'WebcamFilteringWithOpenCVandCUDANPP'
add_subdirectory(WebcamFilteringWithOpenCVandCUDANPP)

# Set default build type if none is specified
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
	message(STATUS "Setting build type to 'RelWithDebInfo' as none was specified.")
	set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "Choose the type of build." FORCE)
endif()

message(STATUS "Root CMakeLists.txt processing complete.")