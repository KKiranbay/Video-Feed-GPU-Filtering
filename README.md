# Video Feed GPU Filtering

## Description
This solution leverages GPU acceleration to apply real-time filtering to video feeds. By utilizing the parallel processing capabilities of modern GPUs, the system achieves high-performance video processing, enabling smooth and efficient application of complex filters.

### Dependencies
- Visual Studio 2022
- vcpkg: 
	- OpenCV
	- imgui
	- sdl2
	- gl3w
- CUDA and NPP

### To build
- install cmake
- download vcpkg, install:
	- set %VCPKG_ROOT% as environment variable
	- set %PATH% so that it includes %VCPKG_ROOT%

- create a "build" folder then call this inside it:
#### Visual Studio 2022 CMake Command
cmake .. -G "Visual Studio 17 2022" -A x64 -DCMAKE_TOOLCHAIN_FILE="%VCPKG_ROOT%/scripts/buildsystems/vcpkg.cmake"
