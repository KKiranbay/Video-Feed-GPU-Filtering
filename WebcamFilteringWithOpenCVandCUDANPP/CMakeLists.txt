# Gather Source and Header Files
file(GLOB_RECURSE PROJECT_SOURCES
    RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
    "*.cpp" "*.cxx" "*.cc"
)
file(GLOB_RECURSE PROJECT_HEADERS
    RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
    "*.h" "*.hpp"
)

source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}" FILES ${PROJECT_HEADERS} ${PROJECT_SOURCES})

# Define the Executable Target
add_executable(${PROJECT_NAME}
    ${PROJECT_SOURCES}
    ${PROJECT_HEADERS}
    ${PROJECT_MISC_FILES}
)

# Target Include Directories
target_include_directories(${PROJECT_NAME} PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/src"
)

# Find and Link Dependencies
find_package(gl3w REQUIRED)
message(STATUS "gl3w found.")

find_package(imgui REQUIRED)
message(STATUS "ImGui found.")

find_package(OpenCV REQUIRED
    COMPONENTS
        core
		highgui
		imgproc
		videoio
        cudaarithm
		cudaimgproc
		
        # cudabgsegm
        # cudafeatures2d
        # cudafilters
        # cudalegacy
        # cudaobjdetect
        # cudaoptflow
        # cudastereo
        # cudawarping
        # cudev
    CONFIG # Use CONFIG mode for OpenCV to ensure it uses the CMake config files from vcpkg
)
message(STATUS "OpenCV found: Version ${OpenCV_VERSION}")
message(STATUS "OpenCV Libraries: ${OpenCV_LIBS}")

find_package(SDL2 REQUIRED)
message(STATUS "SDL2 found: Includes=${SDL2_INCLUDE_DIRS}, Libraries=${SDL2_LIBRARIES}")

# Link Libraries to the Executable
target_link_libraries(${PROJECT_NAME} PRIVATE
	CUDA::cudart
	CUDA::nppial
	CUDA::nppicc
	CUDA::nppif
	unofficial::gl3w::gl3w
	imgui::imgui
	${OpenCV_LIBS}
	SDL2::SDL2
)

# Compile Definitions
# target_compile_definitions(${PROJECT_NAME} PRIVATE SDL_MAIN_HANDLED) -> Already added to main.cpp

message(STATUS "src/CMakeLists.txt processing complete for target ${PROJECT_NAME}.")